import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fire import Fire
from tqdm import tqdm
from bgflow.distribution.normal import NormalDistribution
from bgflow.nn.flow.coupling import SplitFlow, SwapFlow, CouplingFlow, InverseFlow
from bgflow.nn.flow.sequential import SequentialFlow
from bgflow.nn.flow.transformer.affine import AffineTransformer
from bgflow.nn.dense import DenseNet
from bgflow.bg import BoltzmannGenerator

import bgflow as bg

from target import TARGETS_DICT

def train_bgflow(
    n_particles=13,
    dim=39,
    n_training_steps=10000,
    batch_size=128,
    lr=1e-3,
    temperature=1.0,
    output_dir="experiments/bgflow_lj13",
    n_samples_to_save=10000
):
    """
    Trains a Boltzmann Generator (Normalizing Flow) on LJ13 potential
    via energy-based training (KL Divergence minimization).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Define Target Energy (LJ13)
    target_name = "lennard_jones"
    target_class = TARGETS_DICT[target_name]
    target_energy = target_class(
        dim=dim, 
        n_particles=n_particles, 
        eps=2.0, 
        rm=1.0, 
        oscillator=True, 
        oscillator_scale=1.0
    ).to(device)

    # define prior (std normal)
    prior = NormalDistribution(dim).to(device)

    # define flow architecture (RealNVP)
    layers = []
    n_coupling_layers = 6
    
    # Calculate split dimensions for odd-dimensional inputs (39)
    split_dim = dim // 2      # 19
    cond_dim = dim - split_dim # 20

    for _ in range(n_coupling_layers):
        # SplitFlow splits x into (z1, z2). 
        # z2 has dimension `split_dim` (19). 
        # z1 has dimension `dim - split_dim` (20).
        layers.append(SplitFlow(split_dim))
        
        # SwapFlow swaps them -> (z2, z1).
        # We transform z2 (size 19) conditioned on z1 (size 20).
        layers.append(SwapFlow())
        
        # CouplingFlow transforms the first element (z2, size 19).
        # The transformer nets take the second element (z1, size 20) as input.
        layers.append(CouplingFlow(
            AffineTransformer(
                # DenseNet Input: cond_dim (20). Output: split_dim (19)
                shift_transformation=DenseNet([cond_dim, 128, 128, split_dim], activation=torch.nn.ReLU()),
                scale_transformation=DenseNet([cond_dim, 128, 128, split_dim], activation=torch.nn.ReLU())
            )
        ))
        
        # Merge back. InverseFlow(SplitFlow) usually concatenates.
        # Since we swapped, we are concatenating (19, 20) -> 39.
        # This effectively shuffles the dimensions for the next layer.
        layers.append(InverseFlow(SplitFlow(split_dim)))
    
    flow = SequentialFlow(layers).to(device)
    generator = BoltzmannGenerator(prior, flow, target_energy)
    
    # optimization loop for energy-based training/KLL
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    loss_history = []
    
    print("Starting training...")
    pbar = tqdm(range(n_training_steps))
    for i in pbar:
        optimizer.zero_grad()
        
        # Sample z from prior
        z = prior.sample(batch_size)
        
        # Pass through flow to get x (N, 39)
        x, dlogp = flow(z, inverse=False)
        
        # Calculate Target Energy
        energy = target_energy.energy(x)
        
        # Loss = Energy / T - Entropy
        loss = (energy.mean() / temperature) - dlogp.mean()
        
        # Check for NaNs/Infs (common in LJ collisions)
        if torch.isnan(loss) or torch.isinf(loss):
            # pbar.set_description("NaN detected, skipping")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.4f}")

    # plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("KLL Loss")
    plt.title("BGFlow Training Loss")
    plt.savefig(os.path.join(output_dir, "bgflow_loss.png"))
    plt.close()

    # generate and save samples
    print(f"generating {n_samples_to_save} samples...")
    with torch.no_grad():
        samples = generator.sample(n_samples_to_save)
        samples_np = samples.cpu().numpy()
    
    save_path = os.path.join(output_dir, "bgflow_samples_T1.0.npy")
    np.save(save_path, samples_np)
    print(f"samples saved to {save_path}")

if __name__ == "__main__":
    Fire(train_bgflow)