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
        # print(f"Warning: Could not get git hash ({e})")
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
    # MINIMIZING the oracle means we prefer LOW energy
    if mode == "gp_classic":  # LCB (deterministic)
        return mu - beta * std
    elif mode == "greedy":
        return mu
    elif mode == "posterior_sample":  # TS style
        # E ~ N(mu, std)
        # use rsample() to keep the gradient flow (reparameterization trick)
        # which allows the optimizer in high dim mode to work
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
        
        # ensure output dir exists immediately for per-iter plotting
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
                print(f"iter {i}: sampled {x_next.numpy().ravel()} -> y={y_next_obs.item():.4f}")
                if use_grid_sampling: 
                    self._plot_iteration(i, debug_info)

        self._save_results()

    def _setup_grid(self):  # NOTE only for low dim e.g. self.dim <= 2
        self.n_grid = self.cfg['n_grid']
        linspace = torch.linspace(self.bounds[0], self.bounds[1], self.n_grid)
        
        if self.dim == 1:
            self.X_grid = linspace.view(-1, 1)
            self.grid_vol = (self.bounds[1] - self.bounds[0]) / self.n_grid
        else:  # nd grid
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
        # optim loop for high dim
        n_restarts = 10
        candidates = torch.rand(n_restarts, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates.requires_grad = True
        
        optimizer = torch.optim.Adam([candidates], lr=0.1)
        
        for _ in range(50):
            optimizer.zero_grad()
            mu, var = get_gp_posterior(self.X_train, self.y_train, candidates)
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            
            # NOTE: with 'posterior_sample', the energy surface changes slightly 
            # every step due to resampling, acting like SGD noise
            energy = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
            loss = energy.sum()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                candidates.clamp_(self.bounds[0], self.bounds[1])
        
        with torch.no_grad():
            mu, var = get_gp_posterior(self.X_train, self.y_train, candidates)
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            # one final sample for selection
            energy = calculate_energy(mu, std, mode=self.cfg['energy_mode'])
            
            best_idx = torch.argmin(energy)
            x_next = candidates[best_idx].view(1, self.dim)
            
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
        plt.title(f"iter {iter_idx} (energy mode {self.cfg['energy_mode']})")
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

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.cfg['output_dir']
        # os.makedirs(folder, exist_ok=True) # created in init
        
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
    print(f"Saved plot to {img_name}")

if __name__ == "__main__":
    
    config = {
        'git_hash': get_git_short_hash(),
        'output_dir': "gp_active_sampling_loop_1202",
        'seed': 42,
        'dim': 1,
        'bounds': [-2.5, 2.5],
        'n_grid': 5000,
        'n_iterations': 1000,
        'temperature': 0.2,
        'noise_var': 0.1,
        # options: 'gp_classic', 'greedy', 'posterior_sample'
        'energy_mode': 'posterior_sample' 
    }
    
    exp = ActiveLearningExperiment(config)
    exp.run()