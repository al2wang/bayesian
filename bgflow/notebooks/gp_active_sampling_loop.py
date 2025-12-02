import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

# util for config/versioning
def get_git_short_hash():
    try:
        return subprocess.check_output(
            ["git", "describe", "--always"], 
            cwd=Path(__file__).resolve().parent
        ).strip().decode()
    except Exception as e:
        return "git_not_found"

def oracle_function(x):  # x: (Batch, Dim)
    return torch.sum(torch.sin(x) * torch.cos(x), dim=1, keepdim=True)

def rbf_kernel_torch(x1, x2, lengthscale=1.0, variance=1.0):
    sqdist = torch.cdist(x1, x2)**2
    return variance * torch.exp(-0.5 / lengthscale**2 * sqdist)

def get_gp_posterior(X_train, y_train, X_test, lengthscale=1.0, noise_var=0.01):
    N = X_train.shape[0]
    K = rbf_kernel_torch(X_train, X_train, lengthscale)
    K_s = rbf_kernel_torch(X_train, X_test, lengthscale)
    K_ss = rbf_kernel_torch(X_test, X_test, lengthscale)
    K_inv = torch.linalg.inv(K + noise_var * torch.eye(N))   
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s
    var = torch.diag(cov).view(-1, 1)
    return mu, var

def calculate_energy(mu, std, mode="gp_classic", beta=1.0):
    if mode == "gp_classic":
        return mu - beta * std
    elif mode == "greedy":
        return mu
    elif mode == "posterior_sample":
        # use rsample() to keep the gradient flow
        return torch.distributions.Normal(mu, std).rsample()
    else:
        raise ValueError(f"Unknown energy mode: {mode}")

class ActiveLearningExperiment:
    def __init__(self, config):
        self.cfg = config
        torch.manual_seed(config['seed'])
        self.dim = config['dim']
        self.bounds = config['bounds']
        self.X_train = torch.rand(2, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.y_train = oracle_function(self.X_train) + self.cfg['noise_var'] * torch.randn(2, 1)
        self.history = []
        
        os.makedirs(self.cfg['output_dir'], exist_ok=True)

    def run(self):
        print(f"starting exp; dim={self.dim}, energy mode={self.cfg['energy_mode']}")
        use_grid_sampling = self.dim <= 2
        if use_grid_sampling:
            self._setup_grid()
        
        for i in range(self.cfg['n_iterations']):
            x_next = None
            debug_info = {}
            if use_grid_sampling:
                x_next, debug_info = self._step_grid(i)
            else:
                x_next, debug_info = self._step_optimizer(i)
            
            y_next_true = oracle_function(x_next)
            y_next_obs = y_next_true + self.cfg['noise_var'] * torch.randn(1, 1)
            
            self.X_train = torch.cat([self.X_train, x_next], dim=0)
            self.y_train = torch.cat([self.y_train, y_next_obs], dim=0)
            
            step_data = {
                'iteration': i,
                'x_next': x_next.clone(),
                'y_next': y_next_obs.clone(),
                **debug_info
            }
            self.history.append(step_data)
            
            if i % 50 == 0:
                if use_grid_sampling:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()} -> y={y_next_obs.item():.4f}")
                    self._plot_iteration(i, debug_info)
                else:
                    # call the high dim plotter
                    # pass in x_next, the point we just found
                    print(f"iter {i}: sampled {x_next.numpy().ravel()[:3]}... -> y={y_next_obs.item():.4f}")
                    self._plot_high_dim_slices(x_next.detach(), i, dims_to_plot=[0, 1, 2])

        self._save_results()

    def _setup_grid(self):
        self.n_grid = self.cfg['n_grid']
        linspace = torch.linspace(self.bounds[0], self.bounds[1], self.n_grid)
        
        if self.dim == 1:
            self.X_grid = linspace.view(-1, 1)
            self.grid_vol = (self.bounds[1] - self.bounds[0]) / self.n_grid
        else:
            x = torch.linspace(self.bounds[0], self.bounds[1], int(np.sqrt(self.n_grid)))
            grid = torch.meshgrid([x, x], indexing='ij')
            self.X_grid = torch.stack(grid, dim=-1).reshape(-1, 2)
            side_len = (self.bounds[1] - self.bounds[0]) / int(np.sqrt(self.n_grid))
            self.grid_vol = side_len ** 2

    def _step_grid(self, iter_idx):
        mu, var = get_gp_posterior(self.X_train, self.y_train, self.X_grid)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        energy = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
        logits = -energy / self.cfg['temperature']
        probs = torch.nn.functional.softmax(logits, dim=0)
        prob_densities = probs / self.grid_vol
        categorical = torch.distributions.Categorical(probs.squeeze())
        next_idx = categorical.sample()
        x_next = self.X_grid[next_idx].view(1, self.dim)
        
        return x_next, {
            'mu_grid': mu.cpu(),
            'std_grid': std.cpu(),
            'densities_grid': prob_densities.cpu(),
            'X_grid': self.X_grid.cpu()
        }

    def _step_optimizer(self, iter_idx):
        # precompute K_inv ONCE per iteration
        # assume X_train is fixed during the inner optimization loop
        N = self.X_train.shape[0]
        K = rbf_kernel_torch(self.X_train, self.X_train)
        K_inv = torch.linalg.inv(K + self.cfg['noise_var'] * torch.eye(N))
        
        # optimization loop
        n_restarts = 10
        candidates = torch.rand(n_restarts, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates.requires_grad = True
        
        optimizer = torch.optim.Adam([candidates], lr=0.1)
        
        for _ in range(50):
            optimizer.zero_grad()
            
            # manually compute posterior using precomputed K_inv to save time
            # mu = K_s.T * K_inv * y
            # var = K_ss - K_s.T * K_inv * K_s
            K_s = rbf_kernel_torch(self.X_train, candidates)
            K_ss = rbf_kernel_torch(candidates, candidates)
            
            mu = K_s.T @ K_inv @ self.y_train
            
            # efficient diagonal calc
            # cov term: K_s.T @ K_inv @ K_s
            # only need the diagonal of the result for variance
            weighted_Ks = K_inv @ K_s
            cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
            var = torch.diag(K_ss).view(-1, 1) - cov_term
            
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            energy = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
            loss = energy.sum()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                candidates.clamp_(self.bounds[0], self.bounds[1])
        
        with torch.no_grad():
            # final eval
            K_s = rbf_kernel_torch(self.X_train, candidates)
            K_ss = rbf_kernel_torch(candidates, candidates)
            mu = K_s.T @ K_inv @ self.y_train
            weighted_Ks = K_inv @ K_s
            cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
            var = torch.diag(K_ss).view(-1, 1) - cov_term
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            energy = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
            
            best_idx = torch.argmin(energy)
            # CRITICAL FIX: .detach() breaks the graph history
            x_next = candidates[best_idx].view(1, self.dim).detach()
            
        return x_next, {'min_energy_found': energy[best_idx].item()}
    
    def _plot_iteration(self, iter_idx, debug_info):
        X_grid = debug_info['X_grid']
        mu = debug_info['mu_grid']
        std = debug_info['std_grid']
        densities = debug_info['densities_grid']
        truth = oracle_function(X_grid)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(X_grid.numpy(), truth.numpy(), 'k--', label="truth")
        plt.plot(X_grid.numpy(), mu.numpy(), 'b-', label="GP mean")
        plt.fill_between(X_grid.view(-1).numpy(), 
                        (mu - 2*std).view(-1).numpy(), 
                        (mu + 2*std).view(-1).numpy(), 
                        color='blue', alpha=0.2, label="uncertainty")
        plt.scatter(self.X_train.numpy(), self.y_train.numpy(), c='k', marker='x', label="data")
        plt.title(f"iter {iter_idx} (Energy: {self.cfg['energy_mode']})")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(X_grid.numpy(), densities.numpy(), 'r-', linewidth=2, label="p(x)")
        plt.fill_between(X_grid.view(-1).numpy(), 0, densities.view(-1).numpy(), color='red', alpha=0.1)
        plt.ylabel("density")
        plt.xlabel("X")
        plt.legend()
        
        filename = os.path.join(self.cfg['output_dir'], f"iteration_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    # def _plot_high_dim_slices(self, x_center, iter_idx, dims_to_plot=[0, 1, 2]):
    #     # plots 1D slices of the GP posterior centered at x_center.
    #     # x_center: the point selected in this iteration (1, D)

    #     n_plots = len(dims_to_plot)
    #     fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    #     if n_plots == 1: axes = [axes]
        
    #     # create a temporary dense grid for 1 dimension
    #     N_slice = 200
    #     linspace = torch.linspace(self.bounds[0], self.bounds[1], N_slice)
        
    #     # precompute K_inv for efficiency (standard GP inference)
    #     N = self.X_train.shape[0]
    #     K = rbf_kernel_torch(self.X_train, self.X_train)
    #     K_inv = torch.linalg.inv(K + self.cfg['noise_var'] * torch.eye(N))

    #     for i, dim_idx in enumerate(dims_to_plot):
    #         # create the slice input
    #         # start with x_center repeated N_slice times
    #         x_slice = x_center.repeat(N_slice, 1) 
    #         # overwrite the specific dimension with the grid
    #         x_slice[:, dim_idx] = linspace
            
    #         # GP prediction on the slice
    #         # duplicate the manual GP logic here
    #         K_s = rbf_kernel_torch(self.X_train, x_slice)
    #         K_ss = rbf_kernel_torch(x_slice, x_slice)
            
    #         mu = K_s.T @ K_inv @ self.y_train
    #         weighted_Ks = K_inv @ K_s
    #         cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
    #         var = torch.diag(K_ss).view(-1, 1) - cov_term
    #         std = torch.sqrt(torch.clamp(var, min=1e-6))
            
    #         # calculate energy (use greedy/mean for viz stability, or sample)
    #         # visualizing the "mean energy" for interpretability
    #         energy_mean = calculate_energy(mu, std, mode=self.cfg['energy_mode']) 
            
    #         ax = axes[i]
    #         ax.plot(linspace.numpy(), mu.numpy(), 'b-', label="GP mean")
    #         ax.fill_between(linspace.numpy(), 
    #                        (mu - 2*std).flatten().numpy(), 
    #                        (mu + 2*std).flatten().numpy(), 
    #                        color='blue', alpha=0.2)
    #         ax.plot(linspace.numpy(), energy_mean.numpy(), 'r--', label=f"energy {self.cfg['energy_mode']}")
    #         ax.axvline(x_center[0, dim_idx].item(), color='k', linestyle=':', label="Selected x")
    #         ax.set_title(f"slice of dim {dim_idx}")
    #         if i == 0: ax.legend()
            
    #     plt.suptitle(f"iter {iter_idx}: local slices around selected point")
    #     filename = os.path.join(self.cfg['output_dir'], f"slice_iter_{iter_idx:04d}.png")
    #     plt.savefig(filename)
    #     plt.close()

    def _plot_high_dim_slices(self, x_center, iter_idx, dims_to_plot=[0, 1, 2]):
        # plots 2 rows: 
        # row 0: GP posterior vs truth
        # row 1: energy
        n_plots = len(dims_to_plot)
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 6), sharex='col')
        if n_plots == 1:
            axes = axes.reshape(2, 1) # ensure indexing works
            
        N_slice = 200
        linspace = torch.linspace(self.bounds[0], self.bounds[1], N_slice)
        
        N = self.X_train.shape[0]
        K = rbf_kernel_torch(self.X_train, self.X_train)
        K_inv = torch.linalg.inv(K + self.cfg['noise_var'] * torch.eye(N))

        for i, dim_idx in enumerate(dims_to_plot):
            x_slice = x_center.repeat(N_slice, 1) 
            x_slice[:, dim_idx] = linspace
            
            truth = oracle_function(x_slice)
            
            K_s = rbf_kernel_torch(self.X_train, x_slice)
            K_ss = rbf_kernel_torch(x_slice, x_slice)
            
            mu = K_s.T @ K_inv @ self.y_train
            weighted_Ks = K_inv @ K_s
            cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
            var = torch.diag(K_ss).view(-1, 1) - cov_term
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            energy_vis = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
            
            ax_gp = axes[0, i]
            ax_gp.plot(linspace.numpy(), truth.numpy(), 'k--', label="truth")
            ax_gp.plot(linspace.numpy(), mu.numpy(), 'b-', label="GP mean")
            ax_gp.fill_between(linspace.numpy(), 
                           (mu - 2*std).flatten().numpy(), 
                           (mu + 2*std).flatten().numpy(), 
                           color='blue', alpha=0.2)
            ax_gp.axvline(x_center[0, dim_idx].item(), color='k', linestyle=':', label="selected")
            ax_gp.set_ylabel("f(x)")
            ax_gp.set_title(f"dim {dim_idx}")
            if i == 0: ax_gp.legend(loc='best', fontsize='small')
            
            ax_en = axes[1, i]
            ax_en.plot(linspace.numpy(), energy_vis.numpy(), 'r-', label=f"energy: {self.cfg['energy_mode']}")
            ax_en.axvline(x_center[0, dim_idx].item(), color='k', linestyle=':')
            ax_en.set_ylabel("energy")
            ax_en.set_xlabel(f"x_{dim_idx}")
            if i == 0: ax_en.legend(loc='best', fontsize='small')
            
        plt.suptitle(f"iter {iter_idx}: slices through selected x (dims {dims_to_plot})")
        plt.tight_layout()
        filename = os.path.join(self.cfg['output_dir'], f"slice_iter_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.cfg['output_dir']
        
        config_path = os.path.join(folder, f"config_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(self.cfg, f, indent=4, default=str)
            
        data_path = os.path.join(folder, f"data_{timestamp}.pt")
        save_dict = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'history': self.history,
            'config': self.cfg
        }
        torch.save(save_dict, data_path)
        print(f"Experiment saved to {folder}")
        print(f"  Config: {config_path}")
        print(f"  Data:   {data_path}")
        
        if self.dim == 1:
            plot_experiment(data_path, folder)


def plot_experiment(data_path, output_folder):
    print("...... plotting ......")
    data = torch.load(data_path)
    X_train_final = data['X_train']
    history = data['history']
    
    if 'X_grid' not in history[0]:
        print("no grid data found (high d run), skipping dense plots.")
        return

    X_grid = history[0]['X_grid']
    truth = oracle_function(X_grid)
    
    last_step = history[-1]
    iter_idx = last_step['iteration']
    
    mu = last_step['mu_grid']
    std = last_step['std_grid']
    densities = last_step['densities_grid']
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(X_grid.numpy(), truth.numpy(), 'k--', label="truth")
    plt.plot(X_grid.numpy(), mu.numpy(), 'b-', label="GP mean")
    plt.fill_between(X_grid.view(-1).numpy(), 
                    (mu - 2*std).view(-1).numpy(), 
                    (mu + 2*std).view(-1).numpy(), 
                    color='blue', alpha=0.2, label="uncertainty")
    
    plt.scatter(X_train_final.numpy(), oracle_function(X_train_final).numpy(), c='k', marker='x', label="all samples")
    plt.title(f"iter {iter_idx} (Energy: {data['config']['energy_mode']})")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(X_grid.numpy(), densities.numpy(), 'r-', linewidth=2, label="sampling density p(x)")
    plt.fill_between(X_grid.view(-1).numpy(), 0, densities.view(-1).numpy(), color='red', alpha=0.1)
    plt.ylabel("density")
    plt.xlabel("X")
    plt.legend()
    
    img_name = os.path.join(output_folder, "summary_plot.png")
    plt.savefig(img_name)
    plt.close()
    print(f"saved plot to {img_name}")

if __name__ == "__main__":
    
    config = {
        'git_hash': get_git_short_hash(),
        'output_dir': "gp_active_sampling_loop_1202",
        'seed': 42,
        'dim': 10,
        'bounds': [-2.5, 2.5],
        'n_grid': 5000,
        'n_iterations': 1000, 
        'temperature': 0.2,
        'noise_var': 0.1,
        'energy_mode': 'posterior_sample' 
    }
    
    exp = ActiveLearningExperiment(config)
    exp.run()