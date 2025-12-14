import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from fire import Fire

from target import TARGETS_DICT


def compute_pairwise_distances(x, n_particles, n_dims=3):
    """
    Compute all pairwise distances for a batch of states.
    x: (N, n_particles * n_dims) flat tensor
    returns: (N * n_pairs) flat tensor of all distances
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.float()
    x_reshaped = x.view(-1, n_particles, n_dims)
    dists = torch.cdist(x_reshaped, x_reshaped, p=2)
    triu_indices = torch.triu_indices(n_particles, n_particles, offset=1)
    pairwise_dists = dists[:, triu_indices[0], triu_indices[1]]
 
    return pairwise_dists.flatten()


def compute_lj_energy_torch(
        x,
        n_particles=13,
        n_dims=3,
        eps=2.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0
        ):
    
    target_class = TARGETS_DICT["lennard_jones"]
    dim = 39
    target = target_class(
        dim=39, 
        n_particles=13,
        eps=2.0, 
        rm=1.0, 
        oscillator=True,
        oscillator_scale=1.0
    )
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, dim)
    total_energy = target.energy(x).detach().numpy().ravel()

    return total_energy

def load_dataset(path, n_particles=13, label="Data", is_gt=False):
    
    if not path or not os.path.exists(path):
        print(f"Warning: {label} path not found: {path}")
        return None

    print(f"Loading {label} from {path}...")
    
    try:
        X_tensor = None
        
        if path.endswith('.pt') or path.endswith('.pth'):
            data_obj = torch.load(path, map_location='cpu')
            if isinstance(data_obj, dict):
                if 'X_train' in data_obj: X_tensor = data_obj['X_train']
                elif 'x_next' in data_obj: X_tensor = data_obj['x_next']
                else:
                    for k, v in data_obj.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 2:
                            X_tensor = v
                            break
            elif isinstance(data_obj, torch.Tensor):
                X_tensor = data_obj
                
        elif path.endswith('.npy'):
            data_np = np.load(path)
            # main.py logic: torch.tensor(gt_data, dtype=torch.float32)
            X_tensor = torch.tensor(data_np, dtype=torch.float32)
            
        elif path.endswith('.npz'):
            data_npz = np.load(path)
            found_key = None
            for key in ['arr_0', 'data', 'samples']:
                if key in data_npz:
                    found_key = key
                    break
            if not found_key: found_key = list(data_npz.keys())[0]
            X_tensor = torch.tensor(data_npz[found_key], dtype=torch.float32)

        if X_tensor is None:
            print(f"Error: Could not extract tensor from {label}")
            return None

        # NOTE: in main.py .reshape(-1, self.dim) where dim=39
        X_tensor = X_tensor.reshape(-1, n_particles * 3)
        energy = compute_lj_energy_torch(X_tensor, n_particles=n_particles)
        valid_mask = np.isfinite(energy)    # filter NaNs for plotting
        X_valid = X_tensor[torch.tensor(valid_mask)]
        dists_clean = compute_pairwise_distances(X_valid, n_particles).numpy()

        return {
            "name": label,
            "y": energy[valid_mask],
            "dists_clean": dists_clean
        }
        
    except Exception as e:
        print(f"Error processing {label}: {e}")
        return None


def plot_comparison(
    active_path="/home/mila/g/guangyuan.wang/scratch/bayesian/experiments/lennard_jones/44/data_20251213_235428.pt",
    pita_path="/home/mila/g/guangyuan.wang/scratch/bayesian/experiments/pita_lj13/pita_samples.npy",
    bgflow_path="/home/mila/g/guangyuan.wang/scratch/bayesian/experiments/bgflow_lj13/bgflow_samples.npy",
    gt_path="./pita/data/lj13/LJ13_temp_3.0/train_split_LJ13-10000.npy",
    output_dir="comparison_plots_lj13",
    filename="lj13_method_comparison.pdf",
    n_particles=13
):

    data_gt = load_dataset(gt_path, n_particles, "Ground Truth", is_gt=True)
    data_active = load_dataset(active_path, n_particles, "Active Sampler")
    data_bgflow = load_dataset(bgflow_path, n_particles, "BGflow") if bgflow_path else None
    data_pita = load_dataset(pita_path, n_particles, "PITA") if pita_path else None

    if data_gt is None:
        print("Error: Ground Truth data is required.")
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    target_class = TARGETS_DICT["lennard_jones"]
    dim = 39
    target = target_class(
        dim=39, 
        n_particles=13,
        eps=2.0, 
        rm=1.0, 
        oscillator=True,
        oscillator_scale=1.0
    )
    gt_data_this_one = np.load(gt_path)
    X_gt = torch.tensor(gt_data_this_one, dtype=torch.float32).reshape(-1, dim)
    y_gt = target.energy(X_gt).detach().numpy().ravel()

    gt_min, gt_max = y_gt.min(), y_gt.max()
    
    # NOTE: default bounds based on GT
    lower_bound = gt_min - 5
    upper_bound = gt_max + 20   # default padding
    
    # clamp to prevent GT from vanishing
    if data_active is not None:
        y_gen = data_active['y']
        gen_min = y_gen.min()
        gen_p95 = np.percentile(y_gen, 95)
        lower_bound = min(gt_min, gen_min) - 5
        # show at most GT max+1000, or the 95th percentile of generation
        upper_bound = max(gt_max + 50, min(gen_p95, gt_max + 1000))

    print(f"plot range = [{lower_bound:.2f}, {upper_bound:.2f}]")

    # PLOT #1 : POTENTIAL ENERGY
    ax = axes[0]
    
    # ground truth: plot WITHOUT forcing range (auto scale bins to GT data)
    # NOTE: this matches main.py: axes[0].hist(y_gt, bins=50, ...)
    ax.hist(y_gt, bins=50, density=True, alpha=0.6, color='gray', label='Ground Truth (MCMC)')

    # generated data: plot WITH forced range (clipped view)
    # NOTE: this matches main.py: axes[0].hist(..., range=(lower_bound, upper_bound))
    if data_bgflow:
        ax.hist(data_bgflow['y'], bins=50, density=True, alpha=0.5, color='blue',
                range=(lower_bound, upper_bound), label=f"BGflow")
        
    if data_pita:
        ax.hist(data_pita['y'], bins=50, density=True, alpha=0.4, color='green',
                range=(lower_bound, upper_bound), label=f"PITA")

    if data_active:
        ax.hist(data_active['y'], bins=50, density=True, alpha=0.3, color='red',
                range=(lower_bound, upper_bound), label=f"Active Sampler")

    ax.set_xlabel("Potential Energy")
    ax.set_ylabel("Normalized Density")
    ax.set_title(f"Energy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PLOT #2 : INTERATOMIC DISTANCE
    ax = axes[1]
    
    ax.hist(data_gt['dists_clean'], bins=50, density=True, alpha=0.6, color='gray', label='Ground Truth')
    
    if data_bgflow:
        ax.hist(data_bgflow['dists_clean'], bins=50, density=True, alpha=0.5, color='blue', label='BGflow')
    if data_pita:
        ax.hist(data_pita['dists_clean'], bins=50, density=True, alpha=0.4, color='green', label='PITA')
    if data_active:
        ax.hist(data_active['dists_clean'], bins=50, density=True, alpha=0.3, color='red', label='Active Sampler')

    ax.set_xlabel("Interatomic Distance")
    ax.set_ylabel("Normalized Density")
    ax.set_title("Pairwise Distance Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")
    plt.close()

if __name__ == "__main__":
    Fire(plot_comparison)