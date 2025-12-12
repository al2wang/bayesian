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


def compute_pairwise_distances(x, n_particles, n_dims=3):
    """
    compute all pairwise distances for a batch of states.
    x: (N, n_particles * n_dims) flat tensor
    returns: (N * n_pairs) flat tensor of all distances
    """
    x_reshaped = x.view(-1, n_particles, n_dims)    # reshape to (N, particles, coords)
    dists = torch.cdist(x_reshaped, x_reshaped, p=2)    # compute pairwise distance matrix (N, P, P)
    triu_indices = torch.triu_indices(
        n_particles,
        n_particles,
        offset=1
    )   # extract upper triangle indices (excluding diagonal) to avoid duplicates and zeros
    pairwise_dists = dists[:, triu_indices[0], triu_indices[1]] # gather distances, result is (N, n_pairs)
    
    return pairwise_dists.flatten()



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
            self.n_particles = config["lj_n_particles"]            
            # pass physical constants (eps=2.0, rm=1.0)
            self.target = target_class(
                dim=self.dim, 
                n_particles=self.n_particles, 
                eps=2.0, 
                rm=1.0, 
                oscillator=True,
                oscillator_scale=1.0
            )
        else:
            self.target = target_class(dim=self.dim)
            self.n_particles = None

        n_initial = 50
        self.X_train = self.sample_domain_uniform(num=n_initial)
        self.y_train = self.target.energy(self.X_train)
        self.history = []

        self.X_grid = None
        self.rkls_kde = []
        self.rkls_grid = []
        self.log_Z = 0.0
        
        # load ground truth data for plotting (if available)
        self.X_gt = None
        if config["gt_data_path"] and os.path.exists(config["gt_data_path"]):
            try:
                print(f"Loading ground truth data from {config['gt_data_path']}...")
                # assuming .npy file [N_samples, Dim] or [N_samples, Particles, 3]
                gt_data = np.load(config["gt_data_path"])
                self.X_gt = torch.tensor(gt_data, dtype=torch.float32).reshape(-1, self.dim)
                # compute ground truth energies once
                self.y_gt = self.target.energy(self.X_gt).detach().numpy().ravel()
            except Exception as e:
                print(f"failed to load GT data: {e}")
        
        # # if no GT data provided, we cannot plot accurate ground truth comparisons 
        # # for high-D LJ13; initialize placeholders in this case
        # if self.X_gt is None:
        #     print("Warning: No Ground Truth data provided. Using random samples as 'Truth' (inaccurate for LJ13).")
        #     self.X_gt = self.sample_domain_uniform(num=5000)
        #     self.y_gt = self.target.energy(self.X_gt).detach().numpy().ravel()

        # TODO is this really needed?
        self.X_bgflow = None
        if config["bgflow_data_path"] and os.path.exists(config["bgflow_data_path"]):
            try:
                print(f"Loading BGflow baseline data from {config['bgflow_data_path']}...")
                bg_data = np.load(config["bgflow_data_path"])
                # Ensure correct shape (N, dim)
                self.X_bgflow = torch.tensor(bg_data, dtype=torch.float32).reshape(-1, self.dim)
                # Compute energies using OUR potential function to ensure fair comparison
                self.y_bgflow = self.target.energy(self.X_bgflow).detach().numpy().ravel()
            except Exception as e:
                print(f"Failed to load BGflow data: {e}")


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
                # LJ13 Specific Plotting
                if self.cfg["target_name"] == "lennard_jones":
                    print(f"iter {i}: sampled energy y={y_next.item():.4f}")
                    self._plot_lj13_metrics(i)
                
                # Standard Dimensional Plotting
                elif self.dim == 1:
                    self._plot_1d(i, debug_info)
                elif self.dim == 2:
                    self._plot_2d(i, debug_info)
                else:
                    self._plot_high_dim_slices(x_next.detach(), i)
                
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
        if pts_per_dim == 2 and self.dim > 5:
            print("WARNING: Grid resolution is extremely low (2 pts/dim).")

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


    def _plot_lj13_metrics(self, iter_idx):
        # GROUND TRUTH - gray
        # ACTIVE SAMPLES - red
        # BGFLOW SAMPLES - blue

        # plots histograms for (1) potential energy and (2) interatomic distances
        # filters NaN/Inf and clips extreme outliers for visibility

        X_gen = self.X_train.detach() 
        y_gen = self.y_train.detach().numpy().ravel()
        
        valid_indices = np.isfinite(y_gen)
        y_gen_clean = y_gen[valid_indices]
        X_gen_clean = X_gen[torch.tensor(valid_indices)]
        
        if len(y_gen_clean) == 0:
            print("Warning: All generated energies are NaN/Inf. Skipping plot.")
            return

        y_gt = self.y_gt[np.isfinite(self.y_gt)]
        
        # compute distance
        dists_gen = compute_pairwise_distances(X_gen_clean, self.n_particles, n_dims=3).numpy()
        dists_gt = compute_pairwise_distances(self.X_gt, self.n_particles, n_dims=3).numpy()
        



        # prepare bgflow
        y_bg_clean = np.array([])
        dists_bg = np.array([])
        
        if self.X_bgflow is not None:
            y_bg_raw = self.y_bgflow
            valid_bg = np.isfinite(y_bg_raw)
            y_bg_clean = y_bg_raw[valid_bg]
            X_bg_clean = self.X_bgflow[torch.tensor(valid_bg)]
            
            if len(y_bg_clean) > 0:
                dists_bg = compute_pairwise_distances(X_bg_clean, self.n_particles, n_dims=3).numpy()




        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # NOTE: DETERMINE ROBUST RANGE FOR ENERGY
        # Focus on the Ground Truth range, but extend slightly to show near-misses.
        # If generated samples are ALL way off (e.g. > 1e10), we still want to show them 
        # (or show that they are off scale), but we must not let 1e30 ruin the plot for GT.
        # Heuristic: use [min(GT), max(GT) + padding] OR [min(GT), percentile_90(Gen)]
        # We cap the view at a "reasonable" high energy (e.g., 200) or relative to GT range.
        
        gt_min, gt_max = y_gt.min(), y_gt.max()
        gen_min, gen_max = y_gen_clean.min(), y_gen_clean.max()
        
        # Use 95th percentile of GEN data to set upper bound, but clamp to avoid 1e30
        # If all data is 1e30, this percentile will still be high. 
        # So we take min(percentile, gt_max + 500) to keep plot readable.
        
        gen_p95 = np.percentile(y_gen_clean, 95)
        # Dynamic upper bound: Show GT, plus some of the "better" generated samples
        # If gen samples are huge, we cut them off visually so we can see the GT bins.
        upper_bound = max(gt_max + 50, min(gen_p95, gt_max + 1000))
        lower_bound = min(gt_min, gen_min) - 5
        
        # plot #1 potential energy distribution
        axes[0].hist(y_gt, bins=50, density=True, alpha=0.6, color='gray', label='ground truth (MCMC)')
        if len(y_bg_clean) > 0:
            axes[0].hist(y_bg_clean, bins=50, density=True, alpha=0.5, color='blue', 
                         range=(lower_bound, upper_bound),
                         label=f'BGflow (N={len(y_bg_clean)})')
        axes[0].hist(y_gen_clean, bins=50, density=True, alpha=0.4, color='red', 
                     range=(lower_bound, upper_bound),
                     label=f'active samples (N={len(y_gen_clean)})')
        axes[0].set_xlabel("Potential Energy")
        axes[0].set_ylabel("Normalized Density")
        axes[0].set_title(f"Iter {(iter_idx)}: Energy (View Range: [{lower_bound:.1f}, {upper_bound:.1f}])")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # # warn if huge clipping occurred
        # if gen_max > upper_bound:
        #     axes[0].text(0.5, 0.9, f"Max Energy: {gen_max:.1e} (Clipped)", 
        #                  transform=axes[0].transAxes, color='red', ha='center')

        # plot #2 interatomic distance distribution
        axes[1].hist(dists_gt, bins=50, density=True, alpha=0.6, color='gray', label='ground truth')
        if len(dists_bg) > 0:
            axes[1].hist(dists_bg, bins=50, density=True, alpha=0.5, color='blue', label='BGflow')
        axes[1].hist(dists_gen, bins=50, density=True, alpha=0.4, color='red', label='active samples')
        axes[1].set_xlabel("interatomic distance")
        axes[1].set_ylabel("normalized density")
        axes[1].set_title("interatomic distances")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.cfg['output_dir'], f"lj13_metrics_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()


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
        if n_plots == 1: axes = axes.reshape(2, 1) 
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
        target_name="lennard_jones",
        dim=39, # for lj13, 13 particles * 3 dims = 39
        output_dir="experiments",
        seed=100,
        bounds=[-2.5, 2.5],
        n_grid=5000,
        n_iterations=1000, 
        temperature=2.0,
        noise_var =0.2,
        sampling_mode="posterior",
        n_candidates=1000,  # NOTE: use random candidates for high dim; grid fails > 3D
        lj_n_particles=13,
        gt_data_path="./pita/data/lj13/LJ13_temp_3.0/train_split_LJ13-10000.npy",
        bgflow_data_path="experiments/bgflow_lj13/bgflow_samples.npy" 
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
        'gt_data_path': gt_data_path,
        'bgflow_data_path': bgflow_data_path
    }

    print(config)
    exp = ActiveLearningExperiment(config)
    try:
        exp.run()
    except KeyboardInterrupt:
        exp._save_results()

if __name__ == "__main__":
    Fire(main)