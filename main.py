from fire import Fire
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
import math

from target import TARGETS_DICT

# new imports for botorch logic
from botorch.models import SingleTaskGP 
from botorch.fit import fit_gpytorch_mll 
from gpytorch.mlls import ExactMarginalLogLikelihood 
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement 
from botorch.optim import optimize_acqf 

from scipy import stats

# util for config/versioning
def get_git_short_hash():
    try:
        return subprocess.check_output(
            ["git", "describe", "--always"], 
            cwd=Path(__file__).resolve().parent
        ).strip().decode()
    except Exception as e:
        return "git_not_found"

def rbf_kernel_torch(x1, x2, lengthscale=1.0, variance=1.0):
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    sqdist = torch.cdist(x1, x2)**2
    return variance * torch.exp(-0.5 / lengthscale**2 * sqdist)

def get_gp_posterior(X_train, y_train, X_test, lengthscale, noise_var):
    N = X_train.shape[0]
    K = rbf_kernel_torch(X_train, X_train, lengthscale)
    K_s = rbf_kernel_torch(X_train, X_test, lengthscale)
    K_ss = rbf_kernel_torch(X_test, X_test, lengthscale)
    K_inv = torch.linalg.inv(K + noise_var * torch.eye(N))   
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    var = torch.diag(cov).view(-1, 1)
    return mu, var

class ActiveLearningExperiment:
    def __init__(self, config):
        self.cfg = config
        torch.manual_seed(config['seed'])
        self.dim = config['dim']
        self.shape = (self.dim,) 
        self.sampling_mode = config["sampling_mode"]
        self.n_candidates = config["n_candidates"]
        self.use_grid_sampling = self.n_candidates == 0
        self.noise_var = config["noise_var"]
        self.bounds = config['bounds']

        target_class = TARGETS_DICT[config["target_name"]]
        if config["target_name"] == "lennard_jones":
            n_particles = config["lj_n_particles"]
            self.target = target_class(dim=self.dim, n_particles=n_particles)
        else:
            self.target = target_class(dim=self.dim)

        n_initial = 50
        self.X_train = self.sample_domain_uniform(num=n_initial)
        self.y_train = self.target.energy(self.X_train)
        self.history = []

        self.X_grid = None
        self.rkls_kde = []
        self.rkls_grid = []
        self.log_Z = 0.0 # Store log_Z instead of Z for stability
        
        os.makedirs(config['output_dir'], exist_ok=True)

    def sample_domain_uniform(self, num=1):
        return torch.rand(num, *self.shape) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def calculate_energy(self, mu, std, beta=1.0):
        mode = self.sampling_mode
        assert mode in ["pessimistic", "greedy", "posterior"]
        if mode == "pessimistic":
            return mu - beta * std
        elif mode == "greedy":
            return mu
        else:
            return torch.distributions.Normal(mu, std).rsample()

    def run(self):
        self._save_results(only_cfg=True)
        
        if self.use_grid_sampling:
            self._setup_grid()
            self.target_energies = self.target.energy(self.X_grid)
            self.target_densities = self._get_probs(self.target_energies, self.grid_vol)

        # MODIFIED: Get Log Z to prevent underflow with high energies
        self.log_Z = self._get_log_Z()

        for i in range(self.cfg['n_iterations']):
            x_next = None
            debug_info = {}
            
            if self.use_grid_sampling:
                x_next, debug_info = self._step_grid(i)
            else:
                x_next, debug_info = self._step_uniform(i)
            
            y_next = self.target.energy(x_next)
            
            self.X_train = torch.cat([self.X_train, x_next], dim=0)
            self.y_train = torch.cat([self.y_train, y_next], dim=0)
            
            step_data = {
                'iteration': i,
                'x_next': x_next.clone(),
                'y_next': y_next.clone(),
                **debug_info
            }
            self.history.append(step_data)
            
            if i % 10 == 0: 
                if self.dim == 1:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()} -> y={y_next.item():.4f}")
                    self._plot_1d(i, debug_info)
                elif self.dim == 2:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()} -> y={y_next.item():.4f}")
                    self._plot_2d(i, debug_info)
                else:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()[:3]}... -> y={y_next.item():.4f}")
                    self._plot_high_dim_slices(x_next.detach(), i, dims_to_plot=[0, 1] if self.dim > 1 else [0])
                
                self._plot_rkl_loss(i)

        self._save_results()

    def _setup_grid(self):
        pts_per_dim = max(2, int(self.cfg['n_grid'] ** (1 / self.dim)))        
        ranges = [torch.linspace(self.bounds[0], self.bounds[1], pts_per_dim) for _ in range(self.dim)]
        grids = torch.meshgrid(*ranges, indexing='ij')
        self.X_grid = torch.stack(grids, dim=-1).reshape(-1, self.dim)
        self.n_grid = self.X_grid.shape[0]
        side_len = (self.bounds[1] - self.bounds[0]) / (pts_per_dim - 1)
        self.grid_vol = side_len ** self.dim
        
        print(f"setup grid: dim={self.dim}, pts_per_dim={pts_per_dim}, total_pts={self.n_grid}, grid_vol={self.grid_vol:.4e}")
        
        # WARNING for pigeons
        if pts_per_dim == 2 and self.dim > 5:
            print("WARNING: Grid resolution is extremely low (2 pts/dim). High energy collisions guaranteed.")

    def _step_grid(self, iter_idx):
        return self._step_with_candidates(iter_idx, self.X_grid.reshape(-1, *self.shape))

    def _get_probs(self, energy, grid_vol=None):
        shape = energy.shape
        energy = energy.squeeze()
        logits = -energy / self.cfg['temperature']
        probs = torch.nn.functional.softmax(logits, dim=0)
        if grid_vol:
            probs = probs / grid_vol
        return probs.reshape(shape)
    
    def _get_log_probs(self, y, grid_vol):
        # MODIFIED: Use log_Z
        output = -y / self.cfg["temperature"] - self.log_Z
        if grid_vol:
            output -= math.log(grid_vol)
        return output 
    
    def _get_log_Z(self):
        # MODIFIED: Calculate Log Z for stability
        if self.X_grid is None:
            samples = self.sample_domain_uniform(num=10000)
        else:
            samples = self.X_grid.reshape(-1, *self.shape)

        energy = self.target.energy(samples)
        logits = -energy.flatten() / self.cfg['temperature']
        
        # logsumexp(logits) - log(N)
        log_Z = torch.logsumexp(logits, dim=0) - math.log(logits.shape[0])
        print(f"this is log_Z={log_Z.item()}")
        return log_Z

    def _step_uniform(self, iter_idx):
        candidates = self.sample_domain_uniform(num=self.n_candidates)
        return self._step_with_candidates(iter_idx, candidates)

    def _step_with_candidates(self, iter_idx, candidates):
        assert (list(candidates.shape)[1:] == list(self.shape))

        mu, var = get_gp_posterior(self.X_train, self.y_train, candidates, lengthscale=1.0, noise_var=self.noise_var)
        all_candidates = candidates
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        energy = self.calculate_energy(mu, std)
        prob_densities = self._get_probs(energy)
        categorical = torch.distributions.Categorical(prob_densities.squeeze())
        next_idx = categorical.sample()
        x_next = all_candidates[next_idx].view(1, *self.shape) 
        
        return x_next, {
            'mu': mu.cpu(),
            'std': std.cpu(),
            'densities': prob_densities.cpu(),
            'candidates': all_candidates.cpu() if not self.use_grid_sampling else None,
        }

    def _plot_1d(self, iter_idx, debug_info):
        X_grid = self.X_grid
        mu = debug_info['mu']
        std = debug_info['std']
        densities = debug_info['densities']
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(X_grid.numpy(), self.target_energies.numpy(), 'k--', label="truth")
        plt.plot(X_grid.numpy(), mu.numpy(), 'b-', label="gp mean")
        plt.fill_between(X_grid.view(-1).numpy(), 
                        (mu - 2*std).view(-1).numpy(), 
                        (mu + 2*std).view(-1).numpy(), 
                        color='blue', alpha=0.2, label="uncertainty")
        plt.scatter(self.X_train.numpy(), self.y_train.numpy(), c='k', marker='x', label="data")
        plt.title(f"iter {iter_idx} (sampling mode: {self.cfg['sampling_mode']})")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(X_grid.numpy(), densities.numpy(), 'r-', linewidth=2, label="model p(x)")
        plt.fill_between(X_grid.view(-1).numpy(), 0, densities.view(-1).numpy(), color='red', alpha=0.1)
        plt.plot(X_grid.numpy(), self.target_densities.numpy(), 'g--', linewidth=2, label="target p(x)")
        
        x_sampled = self.X_train.detach().cpu().numpy().ravel()
        plt.hist(x_sampled, bins=40, density=True, alpha=0.5, color='purple', 
                 label=f'sampled X (history, N={len(x_sampled)})')
        
        try:
            kde = stats.gaussian_kde(self.X_train.T, bw_method="silverman")
            kde_densities = kde.evaluate(self.X_grid.T)
            plt.plot(X_grid.numpy(), kde_densities, 'b--', linewidth=2, label="empirical kde")
        except:
            pass

        plt.ylabel("density")
        plt.xlabel("x")
        plt.legend()
        
        filename = os.path.join(self.cfg['output_dir'], f"plot_1d_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    def _plot_2d(self, iter_idx, debug_info):        
        if self.X_grid is None: return
        pts_per_dim = int(self.n_grid ** 0.5) 
        X = self.X_grid[:, 0].view(pts_per_dim, pts_per_dim).numpy()
        Y = self.X_grid[:, 1].view(pts_per_dim, pts_per_dim).numpy()
        
        truth = self.target_energies.view(pts_per_dim, pts_per_dim).numpy()
        vmax = np.percentile(truth, 95) 
        
        mu = debug_info['mu'].view(pts_per_dim, pts_per_dim).numpy()
        densities = debug_info['densities'].view(pts_per_dim, pts_per_dim).numpy()
        
        train_x = self.X_train.numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        c1 = axes[0].contourf(X, Y, truth, levels=50, cmap='viridis', vmax=vmax)
        axes[0].scatter(train_x[:, 0], train_x[:, 1], c='white', s=10, edgecolors='black', alpha=0.6)
        axes[0].set_title("true energy")
        plt.colorbar(c1, ax=axes[0])
        
        c2 = axes[1].contourf(X, Y, mu, levels=50, cmap='viridis', vmax=vmax)
        axes[1].scatter(train_x[:, 0], train_x[:, 1], c='white', s=10, edgecolors='black', alpha=0.6)
        axes[1].set_title("GP mean prediction")
        plt.colorbar(c2, ax=axes[1])
        
        c3 = axes[2].contourf(X, Y, densities, levels=50, cmap='magma')
        axes[2].scatter(train_x[:, 0], train_x[:, 1], c='cyan', s=10, edgecolors='black', alpha=0.6)
        axes[2].set_title("sampling density p(x)")
        plt.colorbar(c3, ax=axes[2])
        
        plt.suptitle(f"iter {iter_idx}: 2d energy")
        filename = os.path.join(self.cfg['output_dir'], f"plot_2d_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    def _plot_rkl_loss(self, iter_idx):
        if self.dim > 2 and not self.use_grid_sampling:
             return
        
        try:
            kde = stats.gaussian_kde(self.X_train.reshape(self.X_train.shape[0], -1).T, bw_method="silverman")
            output = 0
            for x, y in zip(self.X_train, self.y_train):  
                # MODIFIED: Use log_Z for robust calculation
                # p_norm = exp(-y - log_Z)
                log_p_norm = -y - self.log_Z
                
                q_val = kde.evaluate(x.reshape(-1, 1).numpy())[0]
                
                if q_val > 1e-100:
                    log_q = np.log(q_val)
                    output += (log_q - log_p_norm.item())
            
            output /= self.y_train.shape[0]
            self.rkls_kde.append((iter_idx, output))

            plt.figure(figsize=(10, 8))
            plt.plot([item[0] for item in self.rkls_kde], [item[1] for item in self.rkls_kde], label="reverse kl (kde)")

            # NOTE assuming a grid
            if self.use_grid_sampling and self.dim == self.shape[0]: 
                x_unique = torch.unique(self.X_train, return_counts=True, dim=0 if self.dim == 1 else 1)
                dist = torch.distributions.Categorical(probs=x_unique[1])
                entropy = dist.entropy()
                rkl_grid = -entropy - torch.mean(self._get_log_probs(self.y_train, self.grid_vol))
                self.rkls_grid.append((iter_idx, rkl_grid))
                plt.plot([item[0] for item in self.rkls_grid], [item[1] for item in self.rkls_grid], label="reverse kl (grid)")

            filename = os.path.join(self.cfg['output_dir'], f"loss_rkl.png")
            plt.legend()
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            pass
        
    def _plot_high_dim_slices(self, x_center, iter_idx, dims_to_plot=[0, 1, 2]):
        n_plots = len(dims_to_plot)
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 6), sharex='col')
        if n_plots == 1:
            axes = axes.reshape(2, 1) 
            
        N_slice = 200
        linspace = torch.linspace(self.bounds[0], self.bounds[1], N_slice)
        
        N = self.X_train.shape[0]
        K = rbf_kernel_torch(self.X_train, self.X_train)
        K_inv = torch.linalg.inv(K + self.noise_var * torch.eye(N))

        for i, dim_idx in enumerate(dims_to_plot):
            x_slice = x_center.repeat(N_slice, 1) 
            x_slice_flat = x_slice.reshape(N_slice, -1)
            x_slice_flat[:, dim_idx] = linspace
            x_slice = x_slice_flat.view(N_slice, *self.shape)
            
            truth = self.target.energy(x_slice)
            
            K_s = rbf_kernel_torch(self.X_train, x_slice)
            K_ss = rbf_kernel_torch(x_slice, x_slice)
            
            mu = K_s.T @ K_inv @ self.y_train
            weighted_Ks = K_inv @ K_s
            cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
            var = torch.diag(K_ss).view(-1, 1) - cov_term
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            energy_vis = self.calculate_energy(mu, std)
            
            ax_gp = axes[0, i]
            ax_gp.plot(linspace.numpy(), truth.numpy(), 'k--', label="truth")
            ax_gp.plot(linspace.numpy(), mu.numpy(), 'b-', label="GP mean")
            ax_gp.fill_between(linspace.numpy(), 
                           (mu - 2*std).flatten().numpy(), 
                           (mu + 2*std).flatten().numpy(), 
                           color='blue', alpha=0.2)
            ax_gp.axvline(x_center.reshape(-1)[dim_idx].item(), color='k', linestyle=':', label="selected")
            ax_gp.set_ylabel("f(x)")
            ax_gp.set_title(f"dim {dim_idx}")
            if i == 0: ax_gp.legend(loc='best', fontsize='small')
            
            ax_en = axes[1, i]
            ax_en.plot(linspace.numpy(), energy_vis.numpy(), 'r-', label=f"mode: {self.cfg['sampling_mode']}")
            ax_en.axvline(x_center.reshape(-1)[dim_idx].item(), color='k', linestyle=':')
            ax_en.set_ylabel("energy")
            ax_en.set_xlabel(f"x_{dim_idx}")
            if i == 0: ax_en.legend(loc='best', fontsize='small')
            
        plt.suptitle(f"iter {iter_idx}: slices through selected x")
        plt.tight_layout()
        filename = os.path.join(self.cfg['output_dir'], f"plot_slice_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    def _save_results(self, only_cfg=False):
        if only_cfg:
            self.cfg["output_dir"] = str(Path(self.cfg["output_dir"])/self.cfg["target_name"])
            c = 1
            while (Path(self.cfg["output_dir"])/str(c)).exists():
                c += 1
            self.cfg["output_dir"] = str(Path(self.cfg["output_dir"])/str(c))
            Path(self.cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.cfg['output_dir']
        
        config_path = os.path.join(folder, f"config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(self.cfg, f, indent=4, default=str)
        
        if only_cfg: return
            
        data_path = os.path.join(folder, f"data_{timestamp}.pt")
        save_dict = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'history': self.history,
            'config': self.cfg
        }
        torch.save(save_dict, data_path)
        print(f"Experiment saved to {folder}")

def main(
        target_name="double_well",
        dim=3, 
        output_dir="experiments",
        seed=42,
        bounds=[-2.5, 2.5],
        n_grid=5000,
        n_iterations=1000, 
        temperature=0.2,
        noise_var =0.1,
        sampling_mode="posterior",
        n_candidates=0,  
        lj_n_particles=13,
    ):
    config = {
        'git_hash': get_git_short_hash(),
        'output_dir': output_dir,
        'seed': seed,
        'dim': dim, 
        'bounds': bounds,
        'n_grid': n_grid,
        'n_iterations': n_iterations, 
        'temperature': temperature,
        'noise_var': noise_var,
        'sampling_mode': sampling_mode,
        'n_candidates': n_candidates, 
        'target_name': target_name,
        'lj_n_particles': lj_n_particles,
    }

    print(config)
    exp = ActiveLearningExperiment(config)
    try:
        exp.run()
    except KeyboardInterrupt:
        exp._save_results()

if __name__ == "__main__":
    Fire(main)