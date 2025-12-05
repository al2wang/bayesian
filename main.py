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
from botorch.models import SingleTaskGP # import botorch gp model
from botorch.fit import fit_gpytorch_mll # helper to fit model
from gpytorch.mlls import ExactMarginalLogLikelihood # loss function for gp
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, LogExpectedImprovement # acquisition functions
from botorch.optim import optimize_acqf # optimization helper

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

def get_gp_posterior(X_train, y_train, X_test, lengthscale=1.0, noise_var=0.01):

    # TODO: implement with botorch

    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)
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

        # b0 = torch.tensor(self.cfg["bounds"][0])
        # b1 = torch.tensor(self.cfg["bounds"][1])
        # if len(b0.shape) == 0:
        #     b0 = torch.zeros(self.dim) + b0
        # if len(b1.shape) == 0:
        #     b1 = torch.zeros(self.dim) + b1
        # assert list(b0.shape) == [self.dim]
        # assert list(b1.shape) == [self.dim]
        # self.bounds = [b0, b1]
        self.bounds = config['bounds']

        target_class = TARGETS_DICT[config["target_name"]]
        if config["target_name"] == "lennard_jones":
            n_particles = config["lj_n_particles"]
            self.target = target_class(dim=self.dim, n_particles=n_particles)
            # self.shape = torch.tensor([self.dim // n_particles, n_particles], dtype=torch.int8)
            self.shape = (n_particles, self.dim // n_particles)
        else:
            self.target = target_class(dim=self.dim)

        n_initial = 50
        print(self.shape, type(self.shape))
        self.X_train = torch.rand(n_initial, *self.shape) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        self.y_train = self.target.energy(self.X_train) + config['noise_var'] * torch.randn(n_initial, 1)
        self.history = []

        self.X_grid = None
        self.rkls_kde = []
        self.rkls_grid = []
        
        # pre-compute ground truth energy distribution for comparison plots
        # sample a large batch to approximate the true distribution of f(x)
        # use linspace for 1d to get accurate density of states (gray line), otherwise use rand
        # if self.dim == 1:
        #     X_gt = torch.linspace(self.bounds[0], self.bounds[1], 10000).view(-1, 1)
        # else:
        #     X_gt = torch.rand(20000, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        # self.y_gt = self.target.energy(X_gt).detach().numpy().ravel()
        
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
            # use rsample() to keep the gradient flow
            return torch.distributions.Normal(mu, std).rsample()

    def run(self):
        self._save_results(only_cfg=True)
        
        if self.use_grid_sampling:
            assert (self.dim <= 2)
            self._setup_grid()
            self.target_energies = self.target.energy(self.X_grid)
            self.target_densities = self._get_probs(self.target_energies, self.grid_vol)

        self.Z = self._get_Z()

        for i in range(self.cfg['n_iterations']):
            x_next = None
            debug_info = {}
            
            # if use_botorch: # branch for botorch acquisition
            #     x_next, debug_info = self._step_acquisition(i) # use new botorch step
            if self.use_grid_sampling:
                x_next, debug_info = self._step_grid(i)
            else:
                # x_next, debug_info = self._step_acquisition(i)
                x_next, debug_info = self._step_uniform(i)
            
            y_next_true = self.target.energy(x_next)
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
            
            if i % 10 == 0: # reduced print frequency for high-d loop
                # plot the spatial view (grid or slices)
                if self.dim == 1:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()} -> y={y_next_obs.item():.4f}")
                    self._plot_1d(i, debug_info)
                    # self._plot_empirical_distribution(i)
                # elif self.dim == 2:  # TODO; double well plotting function
                #     pass
                else:
                    print(f"iter {i}: sampled {x_next.numpy().ravel()[:3]}... -> y={y_next_obs.item():.4f}")
                    # for botorch/high-d, calculate energy manually for slice visualization
                    self._plot_high_dim_slices(x_next.detach(), i, dims_to_plot=[0, 1] if self.dim > 1 else [0])
                
                # print(f'number of new points {sum(1 for item in self.history if item["new"])} / {i + 1}')
                self._plot_rkl_loss(i)

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
        output = -y / self.cfg["temperature"] - torch.log(self.Z)
        if grid_vol:
            output -= math.log(grid_vol)
        return output 
    
    def _get_Z(self):
        
        if self.X_grid == None:
            samples = self.sample_domain_uniform(num=5000)
        else:
            samples = self.X_grid.reshape(-1, *self.shape)

        energy = self.target.energy(samples)
        logits = -energy / self.cfg['temperature']
        Z = torch.mean(torch.exp(logits))
        print(f"this is Z={Z}")
        return Z


    def _step_uniform(self, iter_idx):
        candidates = self.sample_domain_uniform(num=self.n_candidates)
        return self._step_with_candidates(iter_idx, candidates)


    def _step_with_candidates(self, iter_idx, candidates):
        # print(f'shape = {candidates.shape}')

        # sample_new_only=True

        assert (list(candidates.shape)[1:] == list(self.shape))

        print(self.X_train.shape, self.y_train.shape, candidates.shape)
        mu, var = get_gp_posterior(self.X_train, self.y_train, candidates)  # TODO: rewrite get_gp_posterior with botorch
        all_candidates = candidates
        # if not sample_new_only:
            # mu = torch.concat([mu, self.y_train])
            # var = torch.concat([var, torch.zeros(self.y_train.shape)])
            # all_candidates = torch.concat([candidates, self.X_train])
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        energy = self.calculate_energy(mu, std)
        # prob_densities = self._get_probs(energy, self.grid_vol)
        prob_densities = self._get_probs(energy)
        categorical = torch.distributions.Categorical(prob_densities.squeeze())
        next_idx = categorical.sample()
        x_next = all_candidates[next_idx].view(1, *self.shape) # TODO check dimensions
        
        return x_next, {
            'mu': mu.cpu(),
            'std': std.cpu(),
            'densities': prob_densities.cpu(),
            'candidates': all_candidates.cpu(),
            # 'new': True if sample_new_only else next_idx < candidates.shape[0]
        }





    # new function for botorch acquisition

    # def _step_acquisition(self, iter_idx): 
    #     # botorch maximizes, so we negate y (since we want to minimize energy)
    #     train_X = self.X_train # use current training x
    #     train_Y = -self.y_train # negate y for minimization task
        
    #     # fit standard gp using botorch helper
    #     gp = SingleTaskGP(train_X, train_Y) # create botorch gp
    #     mll = ExactMarginalLogLikelihood(gp.likelihood, gp) # marginal log likelihood
    #     fit_gpytorch_mll(mll) # fit hyperparameters
        
    #     # define acquisition function based on config
    #     acq_func = None
    #     if self.cfg['energy_mode'] == 'EI':
    #         acq_func = ExpectedImprovement(model=gp, best_f=train_Y.max()) # classic ei
    #     elif self.cfg['energy_mode'] == 'LogEI':
    #          acq_func = LogExpectedImprovement(model=gp, best_f=train_Y.max()) # log ei for numerical stability
    #     elif self.cfg['energy_mode'] == 'UCB':
    #         acq_func = UpperConfidenceBound(model=gp, beta=0.1) # ucb with fixed beta
            
    #     # optimize acquisition function
    #     bounds = torch.tensor([self.bounds] * self.dim).t() # shape bounds for botorch
    #     candidates, _ = optimize_acqf( # optimize using standard bfgs/adam internal
    #         acq_function=acq_func,
    #         bounds=bounds,
    #         q=1, # batch size 1
    #         return_best_only=False,
    #         num_restarts=self.n_candidates, # restarts for non-convex acq optimization
    #         raw_samples=200, # initialization samples
    #     )
        
    #     # x_next = candidates.detach() # detach from graph
    #     # return x_next, {'acq_val': _} # return candidate
    #     return self._step_with_candidates(
    #         iter_idx, candidates.detach().reshape(-1, self.cfg["dim"]))





    # def _step_optimizer(self, iter_idx):
    #     # precompute K_inv ONCE per iteration
    #     # assume X_train is fixed during the inner optimization loop
    #     N = self.X_train.shape[0]
    #     K = rbf_kernel_torch(self.X_train, self.X_train)
    #     K_inv = torch.linalg.inv(K + self.cfg['noise_var'] * torch.eye(N))
        
    #     # optimization loop
    #     n_restarts = 20
    #     candidates = torch.rand(n_restarts, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
    #     candidates.requires_grad = True
        
    #     optimizer = torch.optim.Adam([candidates], lr=0.05)
        
    #     for _ in range(100):
    #         optimizer.zero_grad()
            
    #         K_s = rbf_kernel_torch(self.X_train, candidates)
    #         K_ss = rbf_kernel_torch(candidates, candidates)
            
    #         mu = K_s.T @ K_inv @ self.y_train
            
    #         weighted_Ks = K_inv @ K_s
    #         cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
    #         var = torch.diag(K_ss).view(-1, 1) - cov_term
            
    #         std = torch.sqrt(torch.clamp(var, min=1e-6))
            
    #         energy = self.calculate_energy(mu, std)
    #         loss = energy.sum()
    #         loss.backward()
    #         optimizer.step()
            
    #         with torch.no_grad():
    #             candidates.clamp_(self.bounds[0], self.bounds[1])
        
    #     with torch.no_grad():
    #         # final eval
    #         K_s = rbf_kernel_torch(self.X_train, candidates)
    #         K_ss = rbf_kernel_torch(candidates, candidates)
    #         mu = K_s.T @ K_inv @ self.y_train
    #         weighted_Ks = K_inv @ K_s
    #         cov_term = torch.sum(K_s * weighted_Ks, dim=0).view(-1, 1)
    #         var = torch.diag(K_ss).view(-1, 1) - cov_term
    #         std = torch.sqrt(torch.clamp(var, min=1e-6))
            
    #         energy = self.calculate_energy(mu, std)
            
    #         best_idx = torch.argmin(energy)
    #         # TODO; FIX: .detach() breaks the graph history
    #         x_next = candidates[best_idx].view(1, self.dim).detach()
            
    #     return x_next, {'min_energy_found': energy[best_idx].item()}
    




    # def _plot_empirical_distribution(self, iter_idx):
    #     if self.dim != 1:
    #         return

    #     # get history of samples
    #     # self.X_train contains all samples collected so far
    #     x_sampled = self.X_train.detach().cpu().numpy().ravel()

    #     plt.figure(figsize=(10, 6))
        
    #     # histogram of sampled locations (history)
    #     plt.hist(x_sampled, bins=40, density=True, alpha=0.5, color='red', 
    #              label=f'sampled X (history, N={len(x_sampled)})')
        
    #     # target density
    #     plt.plot(self.X_grid.numpy(), self.target_densities.numpy(), 'g-', linewidth=2, 
    #              label=f'target density')
        
    #     kde = stats.gaussian_kde(self.X_train.T, bw_method="silverman")
    #     kde_densities = kde.evaluate(self.X_grid.T)
    #     plt.plot(self.X_grid.numpy(), kde_densities, 'b--', linewidth=2, label="empirical kde")
        
    #     plt.xlabel("input space x")
    #     plt.ylabel("prob density")
    #     plt.title(f"iter {iter_idx}")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
        
    #     filename = os.path.join(self.cfg['output_dir'], f"empirical_{iter_idx:04d}.png")
    #     plt.savefig(filename)
    #     plt.close()

    def _plot_1d(self, iter_idx, debug_info):
        X_grid = self.X_grid
        mu = debug_info['mu']
        std = debug_info['std']
        densities = debug_info['densities']
        # truth = self.target.energy(X_grid)

        
        # # calculate true boltzmann density for comparison
        # energy_true = truth.view(-1)
        # logits_true = -energy_true / self.cfg['temperature']
        # probs_true = torch.nn.functional.softmax(logits_true, dim=0)
        # densities_true = probs_true / self.grid_vol
        
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
        # plot model density (red)
        plt.plot(X_grid.numpy(), densities.numpy(), 'r-', linewidth=2, label="model p(x)")
        plt.fill_between(X_grid.view(-1).numpy(), 0, densities.view(-1).numpy(), color='red', alpha=0.1)
        # plot true target density (green)
        plt.plot(X_grid.numpy(), self.target_densities.numpy(), 'g--', linewidth=2, label="target p(x)")
        
        # histogram of sampled locations (history)
        x_sampled = self.X_train.detach().cpu().numpy().ravel()
        plt.hist(x_sampled, bins=40, density=True, alpha=0.5, color='purple', 
                 label=f'sampled X (history, N={len(x_sampled)})')
        
        
        kde = stats.gaussian_kde(self.X_train.T, bw_method="silverman")
        kde_densities = kde.evaluate(self.X_grid.T)
        plt.plot(X_grid.numpy(), kde_densities, 'b--', linewidth=2, label="empirical kde")

        
        
        plt.ylabel("density")
        plt.xlabel("x")
        plt.legend()
        
        filename = os.path.join(self.cfg['output_dir'], f"plot_1d_{iter_idx:04d}.png")
        plt.savefig(filename)
        plt.close()

    def _plot_rkl_loss(self, iter_idx):
        # probs = self._get_probs(self.y_train)

        # if len(self.X_train.shape) == 1:
        #     empirical_probs = torch.histogram(self.X_train)
        # else:
        #     empirical_probs = torch.histogramdd(self.X_train)
        # entropy = torch.distributions.Categorical.entropy(empirical_probs)
        # n = self.y_train.shape[0]
        # cross = torch.sum(torch.log(n * self.grid_vol * self.grid_vol * self.y_train)) / (n * self.grid_vol)
        # return -entropy + cross

        kde = stats.gaussian_kde(self.X_train.T, bw_method="silverman")
        output = 0
        for x, y in zip(self.X_train, self.y_train):  # TODO: can compute everything together, faster
            p = torch.exp(-y) / self.Z
            output += torch.log(kde.evaluate(x) / p)
        output /= self.y_train.shape[0]
        self.rkls_kde.append((iter_idx, output))

        # TODO: cross validation

        plt.figure(figsize=(10, 8))
        plt.plot([item[0] for item in self.rkls_kde], [item[1] for item in self.rkls_kde], label="reverse kl (kde)")

        # NOTE assuming a grid
        if self.use_grid_sampling and self.dim == self.shape[0]:   # TODO: maybe consider the case where self.dim != self.shape
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
            ax_gp.axvline(x_center[0, dim_idx].item(), color='k', linestyle=':', label="selected")
            ax_gp.set_ylabel("f(x)")
            ax_gp.set_title(f"dim {dim_idx}")
            if i == 0: ax_gp.legend(loc='best', fontsize='small')
            
            ax_en = axes[1, i]
            ax_en.plot(linspace.numpy(), energy_vis.numpy(), 'r-', label=f"sampling mode: {self.cfg['sampling_mode']}")
            ax_en.axvline(x_center[0, dim_idx].item(), color='k', linestyle=':')
            ax_en.set_ylabel("energy")
            ax_en.set_xlabel(f"x_{dim_idx}")
            if i == 0: ax_en.legend(loc='best', fontsize='small')
            
        plt.suptitle(f"iter {iter_idx}: slices through selected x (dims {dims_to_plot})")
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
        
        if only_cfg:
            return
            
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
        
        # if self.dim == 1:
        #     self.plot_experiment(data_path, folder)

    # def plot_experiment(self, data_path, output_folder):
    #     print("...... plotting ......")
    #     data = torch.load(data_path)
    #     X_train_final = data['X_train']
    #     history = data['history']
        
    #     # check if grid data exists (low dim)
    #     if 'X_grid' in history[0]:
    #         X_grid = history[0]['X_grid']
    #         truth = self.target.energy(X_grid)
    #         last_step = history[-1]
    #         iter_idx = last_step['iteration']
    #         mu = last_step['mu_grid']
    #         std = last_step['std_grid']
    #         densities = last_step['densities_grid']
            
    #         plt.figure(figsize=(10, 8))
    #         plt.subplot(2, 1, 1)
    #         plt.plot(X_grid.numpy(), truth.numpy(), 'k--', label="truth")
    #         plt.plot(X_grid.numpy(), mu.numpy(), 'b-', label="GP mean")
    #         plt.fill_between(X_grid.view(-1).numpy(), 
    #                         (mu - 2*std).view(-1).numpy(), 
    #                         (mu + 2*std).view(-1).numpy(), 
    #                         color='blue', alpha=0.2, label="uncertainty")
    #         plt.scatter(X_train_final.numpy(), self.target.energy(X_train_final).numpy(), c='k', marker='x', label="all samples")
    #         plt.title(f"iter {iter_idx} (sampling mode: {data['config']['sampling_mode']})")
    #         plt.legend()
    #         plt.subplot(2, 1, 2)
    #         plt.plot(X_grid.numpy(), densities.numpy(), 'r-', linewidth=2, label="sampling density p(x)")
    #         plt.fill_between(X_grid.view(-1).numpy(), 0, densities.view(-1).numpy(), color='red', alpha=0.1)
    #         plt.ylabel("density")
    #         plt.xlabel("X")
    #         plt.legend()
    #         img_name = os.path.join(output_folder, "summary_plot.png")
    #         plt.savefig(img_name)
    #         plt.close()
        
    #     # also save the final empirical distribution
    #     y_sampled = data['y_train'].numpy().ravel()
    #     # re-compute ground truth for the summary plot (simple random)
    #     bounds = data['config']['bounds']
    #     dim = data['config']['dim']
    #     X_gt = torch.rand(20000, dim) * (bounds[1] - bounds[0]) + bounds[0]
    #     y_gt = self.target.energy(X_gt).numpy().ravel()
        
    #     # plt.figure(figsize=(8, 6))
    #     # plt.hist(X_gt, bins=50, density=True, alpha=0.5, color='gray', label='Ground Truth')
    #     # plt.hist(y_sampled, bins=30, density=True, alpha=0.6, color='red', label='sampled')
    #     # plt.title("Final Empirical Distribution")
    #     # plt.legend()
    #     # plt.savefig(os.path.join(output_folder, "final_distribution.png"))
    #     # plt.close()

def main(
        target_name="toy_sin_cos",
        dim=1, # TODO use dim 5 for high d testing
        output_dir="experiments",
        seed=42,
        bounds=[-2.5, 2.5],
        n_grid=5000,
        n_iterations=1000, 
        temperature=0.2,
        noise_var =0.1,
        sampling_mode="posterior",
        n_candidates=0,  # n_candidates = 0 means using grid
        lj_n_particles=13,
    ):
    config = {
        'git_hash': get_git_short_hash(),
        'output_dir': output_dir,
        'seed': seed,
        'dim': dim, # TODO use dim 5 for high d testing
        'bounds': bounds,
        'n_grid': n_grid,
        'n_iterations': n_iterations, 
        'temperature': temperature,
        'noise_var': noise_var,
        'sampling_mode': sampling_mode,
        'n_candidates': n_candidates,  # n_candidates = 0 means using grid
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