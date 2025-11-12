# Converted from notebooks/hartmann6d.ipynb
# Magics (%load_ext, %autoreload, %pdb) were removed or converted to comments.

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# first define system dimensionality and a target energy/distribution

# use Hartmann 6-dimensional energy (common benchmark)
dim = 6

from bgflow import Energy
import math


class Hartmann6d(Energy):
	""" Hartmann 6-dimensional function implemented as an Energy."""
	def __init__(self):
		super().__init__(6)
		# coefficients (alpha)
		self.register_buffer('_alpha', torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.get_default_dtype()))
		# matrix A (4 x 6)
		A = [[10.0, 3.0, 17.0, 3.50, 1.7, 8.0],
			 [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
			 [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
			 [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]]
		P = [[1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
			 [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
			 [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
			 [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0]]
		# convert to tensors and scale P by 1e-4 as in the standard definition
		self.register_buffer('_A', torch.tensor(A, dtype=torch.get_default_dtype()))
		self.register_buffer('_P', torch.tensor(P, dtype=torch.get_default_dtype()) * 1e-4)

	def _energy(self, x):
		"""Compute Hartmann function values.
		x: (N,6) tensor -> returns (N,) tensor
		"""
		# ensure shape (N,6)
		orig_shape = x.shape
		if x.dim() == 1:
			x = x.view(1, -1)
		# move A,P to same dtype/device as x
		A = self._A.to(x)
		P = self._P.to(x)
		alpha = self._alpha.to(x)
		# x: (N,6) -> compute (N,4) inner sums
		# diff: (N,4,6)
		diff = (x.unsqueeze(1) - P.unsqueeze(0)) ** 2
		# multiply by A and sum over last dim -> (N,4)
		inner = (A.unsqueeze(0) * diff).sum(dim=-1)
		exps = torch.exp(-inner)
		vals = - (alpha.unsqueeze(0) * exps).sum(dim=-1)
		# return as 1D tensor of length N (or scalar if single input)
		if orig_shape[0] == 6 and len(orig_shape) == 1:
			return vals.view(-1)[0:1].view(())
		return vals


# instantiate the target energy
target = Hartmann6d()


# define some plotting functions

from bgflow.utils.types import assert_numpy

def plot_energy(energy, extent=(-2.5, 2.5), resolution=100, dim=2):
	""" Plot energy functions in 2D """
	xs = torch.meshgrid([torch.linspace(*extent, resolution) for _ in range(2)])
	xs = torch.stack(xs, dim=-1).view(-1, 2)
	xs = torch.cat([
		xs,
		torch.Tensor(xs.shape[0], dim - xs.shape[-1]).zero_()
	], dim=-1)
	us = energy.energy(xs).view(resolution, resolution)
	us = torch.exp(-us)
	plt.imshow(assert_numpy(us).T, extent=extent * 2)
	plt.xlim=(extent[0], extent[1])
	plt.ylim=(extent[0], extent[1])


def plot_samples(samples, weights=None, range=None):
	""" Plot sample histogram in 2D """
	samples = assert_numpy(samples)
	plt.hist2d(
		samples[:, 0], 
		-samples[:, 1],
		weights=assert_numpy(weights) if weights is not None else weights,
		bins=100,
		norm=mpl.colors.LogNorm(),
		range=range,
	)

def plot_bg(bg, target, n_samples=10000, range=[-2.5, 2.5], dim=2):
	""" Plot target energy, bg energy and bg sample histogram"""
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 3, 1)
	plot_energy(target, extent=range, dim=dim)
	plt.title("Target energy")
	plt.subplot(1, 3, 2)
	plot_energy(bg, extent=range, dim=dim)
	plt.title("BG energy")
	plt.subplot(1, 3, 3)
	plot_samples(bg.sample(n_samples), range=[range, range])
	plt.title("BG samples")

# TODO: n_samples=100000 and n_bins=100 originally
def plot_weighted_energy_estimate(bg, target, n_samples=10000, extent=None, n_bins=100, range=[-2, 2], dim=dim, chunk_size=2000):
	""" Plot weighed energy from samples.

	Notes:
	- Defaults reduced to n_samples=10k to lower peak memory usage.
	- We explicitly squeeze/flatten weights before passing them to matplotlib to avoid shape mismatches.
	"""
	# If n_samples is large, use a chunked estimator to avoid storing all samples/weights
	def _chunked_weighted_hist(bg, coord=0, n_samples=10000, chunk=2000, n_bins=100, range=(-2, 2)):
		"""Compute weighted histogram for a single coordinate using chunked sampling.

		Returns bin_centers, density_estimate
		"""
		bins = np.linspace(range[0], range[1], n_bins + 1)
		total_counts = np.zeros(n_bins, dtype=np.float64)
		total_weight = 0.0

		samples_needed = int(n_samples)
		while samples_needed > 0:
			take = min(chunk, samples_needed)
			samples_c, latent_c, dlogp_c = bg.sample(take, with_latent=True, with_dlogp=True)
			# detach to avoid requiring grad when converting to numpy
			log_w = bg.log_weights_given_latent(samples_c, latent_c, dlogp_c).detach().cpu().numpy().astype(np.float64)
			w = np.exp(log_w)
			s0 = samples_c[:, coord].detach().cpu().numpy().astype(np.float64)

			counts, _ = np.histogram(s0, bins=bins, weights=w)
			total_counts += counts
			total_weight += w.sum()
			samples_needed -= take

		bin_width = bins[1] - bins[0]
		density = total_counts / (total_weight * bin_width)
		centers = (bins[:-1] + bins[1:]) / 2.0
		return centers, density

	plt.figure(figsize=(12, 4))

	# For the unweighted histogram and target curve we sample a small subset to avoid high memory use
	_ret = bg.sample(min(5000, n_samples))
	# bg.sample may return samples or (samples, latent, dlogp) depending on flags
	samples_small = _ret[0] if isinstance(_ret, tuple) else _ret
	_, bins, _ = plt.hist(assert_numpy(samples_small[:, 0]), histtype="step", log=True, bins=n_bins, weights=None, density=True, label="samples", range=range)
	xs = torch.linspace(*range, n_bins).view(-1, 1)
	xs = torch.cat([xs, torch.zeros(xs.shape[0], dim - 1)], dim=-1).view(-1, dim)
	us = target.energy(xs).view(-1)
	us = torch.exp(-us)
	us = us / torch.sum(us * (bins[-1] - bins[0]) / n_bins)
	plt.plot(xs[:, 0], us, label="$\\log p(x)$")
	plt.xlabel("$x0$")
	plt.ylabel("log density")
	plt.legend()
	plt.title("unweighed energy")
    
	plt.subplot(1, 3, 2)
	# choose chunked estimator when n_samples is large
	if n_samples > chunk_size:
		centers, density = _chunked_weighted_hist(bg, coord=0, n_samples=n_samples, chunk=chunk_size, n_bins=n_bins, range=range)
		plt.plot(centers, density, drawstyle='steps-mid', label='weighted')
	else:
		samples, latent, dlogp = bg.sample(n_samples, with_latent=True, with_dlogp=True)
		weights_np = assert_numpy(bg.log_weights_given_latent(samples, latent, dlogp).exp().detach()).ravel()
		_, bins, _ = plt.hist(assert_numpy(samples[:, 0]), histtype="step", log=True, bins=n_bins, weights=weights_np, density=True, label="samples", range=range)
		plt.plot(xs[:, 0], us, label="$\\log p(x)$")
	plt.xlabel("$x0$")
	plt.legend()
	plt.title("weighed energy")

	plt.subplot(1, 3, 3)
	plt.xlabel("$x0$")
	plt.ylabel("$x1$")
	# plot a small visualization sample with weights (avoid passing huge arrays to plotting)
	vis_n = min(2000, n_samples)
	vis_samples, vis_latent, vis_dlogp = bg.sample(vis_n, with_latent=True, with_dlogp=True)
	vis_weights = assert_numpy(bg.log_weights_given_latent(vis_samples, vis_latent, vis_dlogp).exp().detach()).ravel()
	plot_samples(vis_samples, weights=vis_weights, range=[range, range])
	plt.title("weighed samples")


# plot target energy (uncomment to run)
# plot_energy(target, dim=dim)


# define a MCMC sampler to sample from the target energy
from bgflow import GaussianMCMCSampler

init_state = torch.Tensor([[-2, 0], [2, 0]])
init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
target_sampler = GaussianMCMCSampler(target, init_state=init_state)


# sample some data (reduced to avoid high memory)
data = target_sampler.sample(5000)
plot_samples(data)


print("reached here")

from bgflow import Energy, Sampler
from torch.distributions.chi2 import Chi2
from torch.distributions.gamma import Gamma


class HypersphericalPrior(Energy, Sampler):
    
	def __init__(self, dim, concentration=1.):
		super().__init__(dim)
		r = np.sqrt(dim) / 2
		rate = concentration / r
		self._gamma = Gamma(concentration, rate)
    
	def _energy(self, x):
		d2 = x.pow(2).sum(dim=-1, keepdim=True)
		d = (d2 + 1e-7).sqrt()
		return -self._gamma.log_prob(d)
        
    
	def _sample(self, n_samples):
		x = torch.Tensor(n_samples, self._dim).normal_()
		d2 = x.pow(2).sum(dim=-1, keepdim=True)
		d = (d2 + 1e-7).sqrt()
		r = x / d
		s = self._gamma.sample((n_samples, 1))
		#         print(s)
		return r * s


# now set up a prior
from bgflow import NormalDistribution

prior = NormalDistribution(dim)
# prior = HypersphericalPrior(dim, concentration=10)


# define a flow with RNVP coupling layers
from bgflow.nn import (
	DenseNet,
	SequentialFlow, 
	CouplingFlow, 
	AffineFlow, 
	SplitFlow, 
	InverseFlow, 
	SwapFlow,
	AffineTransformer
)



# here we aggregate all layers of the flow
layers = []

# start with a splitting layer which splits the input tensor into two 
# flow channels with tensors of half dimensionality
layers.append(SplitFlow(dim // 2))


# now add coupling layers
n_coupling_layers = 4
for _ in range(n_coupling_layers):
    
	# we need to swap dimensions for the mixing
	layers.append(SwapFlow())
    
	# now set up a coupling block
	layers.append(CouplingFlow(
		# we use a affine transformation to transform the RHS conditioned on the LHS
		AffineTransformer(
			# use simple dense nets for the affine shift/scale
			shift_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU()), 
			scale_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU())
		)
	))
    
# finally, we have to merge the two channels again into one tensor
layers.append(InverseFlow(SplitFlow(dim // 2)))
    
# now define the flow as a sequence of all operations stored in layers
flow = SequentialFlow(layers)


# having a flow and a prior, we can now define a Boltzmann Generator
from bgflow import BoltzmannGenerator

bg = BoltzmannGenerator(prior, flow, target)


# initial bg should not be useful
# %pdb  # notebook magic - ignored in script
# plot_bg(bg, target, dim=dim)

# plot_weighted_energy_estimate(bg, target)


from bgflow.utils.types import is_list_or_tuple

class LossReporter:
	"""
		Simple reporter use for reporting losses and plotting them.
	"""
    
	def __init__(self, *labels):
		self._labels = labels
		self._n_reported = len(labels)
		self._raw = [[] for _ in range(self._n_reported)]
    
	def report(self, *losses):
		assert len(losses) == self._n_reported
		for i in range(self._n_reported):
			self._raw[i].append(assert_numpy(losses[i]))
    
	def plot(self, n_smooth=10):
		fig, axes = plt.subplots(self._n_reported, sharex=True)
		if not isinstance(axes, np.ndarray):
			axes = [axes]
		fig.set_size_inches((8, 4 * self._n_reported), forward=True)
		for i, (label, raw, axis) in enumerate(zip(self._labels, self._raw, axes)):
			raw = assert_numpy(raw).reshape(-1)
			kernel = np.ones(shape=(n_smooth,)) / n_smooth
			smoothed = np.convolve(raw, kernel, mode="valid")
			axis.plot(smoothed)
			axis.set_ylabel(label)
			if i == self._n_reported - 1:
				axis.set_xlabel("Iteration")
                
	def recent(self, n_recent=1):
		return np.array([raw[-n_recent:] for raw in self._raw])


# initial training with likelihood maximization on data set
from bgflow.utils.train import IndexBatchIterator

n_batch = 32
batch_iter = IndexBatchIterator(len(data), n_batch)

optim = torch.optim.Adam(bg.parameters(), lr=5e-3)

n_epochs = 5
n_report_steps = 50

reporter = LossReporter("NLL")


for epoch in range(n_epochs):
	for it, idxs in enumerate(batch_iter):
		batch = data[idxs]
        
		optim.zero_grad()
        
		# negative log-likelihood of the batch is equal to the energy of the BG
		nll = bg.energy(batch).mean()
		nll.backward()
        
		reporter.report(nll)
        
		optim.step()
        
		if it % n_report_steps == 0:
			print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4}".format(
					epoch,
					it,
					len(batch_iter),
					*reporter.recent(1).ravel()
				), end="")


reporter.plot()


# bg after ML training
plot_bg(bg, target, n_samples=1000, dim=dim)

plot_weighted_energy_estimate(bg, target, dim=dim)


# train with convex mixture of NLL and KL loss
from bgflow.utils.train import IndexBatchIterator

n_kl_samples = 128
n_batch = 128
batch_iter = IndexBatchIterator(len(data), n_batch)

optim = torch.optim.Adam(bg.parameters(), lr=5e-3)

n_epochs = 5
n_report_steps = 50

# mixing parameter
lambdas = torch.linspace(1., 0.5, n_epochs)

reporter = LossReporter("NLL", "KLL")

torch.linspace(1., 0.5, n_epochs)


for epoch, lamb in enumerate(lambdas):
	for it, idxs in enumerate(batch_iter):
		batch = data[idxs]
        
		optim.zero_grad()
        
		# negative log-likelihood of the batch is equal to the energy of the BG
		nll = bg.energy(batch).mean()
        
		# aggregate weighted gradient
		(lamb * nll).backward()
        
		# kl divergence to the target
		kll = bg.kldiv(n_kl_samples).mean()

		# aggregate weighted gradient
		((1. - lamb) * kll).backward()
        
		reporter.report(nll, kll)
        
		optim.step()
        
		if it % n_report_steps == 0:
			print("\repoch: {0}, iter: {1}/{2}, lambda: {3}, NLL: {4:.4}, KLL: {5:.4}".format(
					epoch,
					it,
					len(batch_iter),
					lamb,
					*reporter.recent(1).ravel()
				), end="")


reporter.plot()


# bg after ML + KL training
plot_bg(bg, target, dim=dim)

plot_weighted_energy_estimate(bg, target, dim=dim)

