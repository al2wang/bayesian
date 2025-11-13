import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# --- GP setup ---
N = 50
X = np.linspace(-2.5, 2.5, N)[:, None]  # shape (N, 1)

def rbf_kernel(X, lengthscale=1.0, variance=1.0):
	X = np.asarray(X)
	sqdist = np.sum(X**2, 1).reshape(-1, 1) + np.sum(X**2, 1) - 2 * np.dot(X, X.T)
	return variance * np.exp(-0.5 / lengthscale**2 * sqdist)

K = rbf_kernel(X, lengthscale=1.0, variance=1.0)
K_inv = np.linalg.inv(K + 1e-6 * np.eye(N))
y = np.sin(X).ravel() + 0.1 * np.random.randn(N)
sigma2 = 0.01

class GPEnergy:
    def __init__(self, y, K_inv, sigma2):
        self.y = torch.tensor(y, dtype=torch.float32)
        self.K_inv = torch.tensor(K_inv, dtype=torch.float32)
        self.sigma2 = sigma2
        self.event_shapes = [(len(y),)]  # 

    def energy(self, f, temperature=None, **kwargs):
        if f.ndim == 1:
            f = f.unsqueeze(0)
        diff = self.y - f
        quad = torch.einsum('bi,ij,bj->b', f, self.K_inv, f)
        data_fit = torch.sum(diff**2, dim=-1) / (2 * self.sigma2)
        return data_fit + 0.5 * quad

dim = N
target = GPEnergy(y, K_inv, sigma2)

# define some plotting functions
from bgflow.utils.types import assert_numpy
def plot_energy(energy, extent=(-2.5, 2.5), resolution=100, dim=2, img_name=None):
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
	plt.xlim(extent[0], extent[1])
	plt.ylim(extent[0], extent[1])
	if img_name:
		plt.savefig(img_name)
		plt.close()

def plot_samples(samples, weights=None, range=None, img_name=None):
	""" Plot sample histogram in 2D """
	samples = assert_numpy(samples)
	plt.hist2d(
		samples[:, 0],
		-samples[:, 1],
		weights=assert_numpy(weights) if weights is not None else weights,
		bins=100,
		norm=mpl.colors.LogNorm(),
		range=range
	)
	if img_name:
		plt.savefig(img_name)
		plt.close()

def plot_bg(bg, target, n_samples=10000, range=[-2.5, 2.5], dim=2, img_prefix=None):
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
	if img_prefix:
		plt.savefig(img_prefix + "_bg.png")
		plt.close()

def plot_weighted_energy_estimate(bg, target, n_samples=100000, extent=None, n_bins=100, range=[-2, 2], dim=dim, img_prefix=None):
	""" Plot weighed energy from samples """
	samples, latent, dlogp = bg.sample(n_samples, with_latent=True, with_dlogp=True)
	log_weights = bg.log_weights_given_latent(samples, latent, dlogp)
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 3, 1)
	_, bins, _ = plt.hist(assert_numpy(samples[:, 0]), histtype="step", log=True, bins=n_bins, weights=None, density=True, label="samples", range=range)
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
	_, bins, _ = plt.hist(assert_numpy(samples[:, 0]), histtype="step", log=True, bins=n_bins, weights=assert_numpy(log_weights.exp()), density=True, label="samples", range=range)
	plt.plot(xs[:, 0], us, label="$\\log p(x)$")
	plt.xlabel("$x0$")
	plt.legend()
	plt.title("weighed energy")
	plt.subplot(1, 3, 3)
	plt.xlabel("$x0$")
	plt.ylabel("$x1$")
	plot_samples(samples, weights=log_weights.exp(), range=[range, range])
	plt.title("weighed samples")
	if img_prefix:
		plt.savefig(img_prefix + "_weighted.png")
		plt.close()

# plot target energy
plot_energy(target, dim=dim, img_name="images/target_energy.png")

# define a MCMC sampler to sample from the target energy
from bgflow import GaussianMCMCSampler
init_state = torch.Tensor([[-2, 0], [2, 0]])
init_state = torch.cat([init_state, torch.Tensor(init_state.shape[0], dim-2).normal_()], dim=-1)
target_sampler = GaussianMCMCSampler(target, init_state=init_state)

# sample some data
data = target_sampler.sample(50000)
plot_samples(data, img_name="images/target_samples.png")

data.pow(2).sum(dim=-1).sqrt().mean()

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
layers = []
layers.append(SplitFlow(dim // 2))
n_coupling_layers = 4
for _ in range(n_coupling_layers):
	layers.append(SwapFlow())
	layers.append(CouplingFlow(
		AffineTransformer(
			shift_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU()),
			scale_transformation=DenseNet([dim // 2, 64, 64, dim // 2], activation=torch.nn.ReLU())
		)
	))
layers.append(InverseFlow(SplitFlow(dim // 2)))
flow = SequentialFlow(layers)

# having a flow and a prior, we can now define a Boltzmann Generator
from bgflow import BoltzmannGenerator
bg = BoltzmannGenerator(prior, flow, target)

# initial bg should not be useful
# plot_bg(bg, target, dim=dim, img_prefix="images/initial")
# plot_weighted_energy_estimate(bg, target, img_prefix="images/initial")

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
	def plot(self, n_smooth=10, img_name=None):
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
		if img_name:
			plt.savefig(img_name)
			plt.close()
	def recent(self, n_recent=1):
		return np.array([raw[-n_recent:] for raw in self._raw])

# initial training with likelihood maximization on data set
from bgflow.utils.train import IndexBatchIterator
n_batch = 8 #. 32
batch_iter = IndexBatchIterator(len(data), n_batch)
optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
n_epochs = 5
n_report_steps = 50
reporter = LossReporter("NLL")

for epoch in range(n_epochs):
	for it, idxs in enumerate(batch_iter):
		batch = data[idxs]
		optim.zero_grad()
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

reporter.plot(img_name="images/loss_nll.png")

# bg after ML training
# plot_bg(bg, target, dim=dim, img_prefix="images/ml")
# plot_weighted_energy_estimate(bg, target, dim=dim, img_prefix="images/ml")





def plot_gp_posterior_samples(bg, X, n_samples=50, img_name=None):
    samples = bg.sample(n_samples).detach().cpu().numpy()  # shape (n_samples, N)
    plt.figure(figsize=(8, 5))
    for i in range(n_samples):
        plt.plot(X.ravel(), samples[i], color='blue', alpha=0.3)
    plt.plot(X.ravel(), y, color='black', lw=2, label='Observed y')
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.title('GP Posterior Samples (BG)')
    plt.legend()
    if img_name:
        plt.savefig(img_name)
        plt.close()
    else:
        plt.show()

def plot_gp_mean_std(bg, X, n_samples=100, img_name=None):
    samples = bg.sample(n_samples).detach().cpu().numpy()
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(X.ravel(), mean, color='red', lw=2, label='Posterior mean')
    plt.fill_between(X.ravel(), mean-2*std, mean+2*std, color='red', alpha=0.2, label='±2 std')
    plt.plot(X.ravel(), y, color='black', lw=2, label='Observed y')
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.title('GP Posterior Mean and Uncertainty')
    plt.legend()
    if img_name:
        plt.savefig(img_name)
        plt.close()
    else:
        plt.show()

plot_gp_posterior_samples(bg, X, n_samples=50, img_name='images/gp_posterior_samples.png')
plot_gp_mean_std(bg, X, n_samples=100, img_name='images/gp_mean_std.png')




# train with convex mixture of NLL and KL loss
from bgflow.utils.train import IndexBatchIterator
n_kl_samples = 128 # 128
n_batch = 128 #. 128
batch_iter = IndexBatchIterator(len(data), n_batch)
optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
n_epochs = 5
n_report_steps = 50
lambdas = torch.linspace(1., 0.5, n_epochs)
reporter = LossReporter("NLL", "KLL")

for epoch, lamb in enumerate(lambdas):
	for it, idxs in enumerate(batch_iter):
		batch = data[idxs]
		optim.zero_grad()
		nll = bg.energy(batch).mean()
		(lamb * nll).backward()
		kll = bg.kldiv(n_kl_samples).mean()
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

reporter.plot(img_name="images/loss_nll_kll.png")

# bg after ML + KL training
# plot_bg(bg, target, dim=dim, img_prefix="images/ml_kl")
# plot_weighted_energy_estimate(bg, target, dim=dim, img_prefix="images/ml_kl")
