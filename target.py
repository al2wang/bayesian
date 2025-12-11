import torch
from bgflow.distribution.energy.double_well import DoubleWellEnergy
from bgflow.distribution.energy.lennard_jones import LennardJonesPotential
from bgflow.distribution.energy.base import Energy


# class DoubleWellEnergy(Energy):
#     def __init__(self, dim, a=0, b=-4.0, c=1.0):
#         super().__init__(dim)
#         self._a = a
#         self._b = b
#         self._c = c

#     def _energy(self, x):
#         d = x[..., [0]]
#         v = x[..., 1:]
#         e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
#         e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
#         return e1 + e2


class ToySinCosEnergy(Energy):
    def __init__(self, dim=1):
        super().__init__(dim)
    
    def _energy(self, x):  # x: (Batch, Dim)
        return torch.sum(torch.sin(x) * torch.cos(x), dim=1, keepdim=True)
    



class WrappedDoubleWell(DoubleWellEnergy):
    def __init__(self, dim=2, a=0, b=-4.0, c=1.0):
        super().__init__(dim=dim, a=a, b=b, c=c)

class WrappedLennardJones(LennardJonesPotential):
    def __init__(self, dim, n_particles=None, **kwargs):
        if n_particles is None:
            n_particles = dim // 2 # assumption if not provided
        
        # MODIFIED: two_event_dims=False
        # This ensures the Energy class treats input as flat (N, dim) rather than (N, particles, space)
        # and handles the reshaping internally or allows the GP to pass flat vectors without error.
        super().__init__(n_particles=n_particles, dim=dim, two_event_dims=False, **kwargs)




TARGETS_DICT = {
    "toy_sin_cos": ToySinCosEnergy,
    "double_well": WrappedDoubleWell,
    "lennard_jones": WrappedLennardJones
}