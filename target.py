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
        # bgflow double well is inherently 2d or higher, usually visualized in 2d
        super().__init__(dim=dim, a=a, b=b, c=c)

class WrappedLennardJones(LennardJonesPotential):
    def __init__(self, dim, n_particles=None, **kwargs):
        # if dim is passed but n_particles not, infer particles (assuming 2d or 3d world?)
        # usually lj is 3d, so dim = n_particles * 3
        if n_particles is None:
            # fallback assumption: 2d space
            n_particles = dim // 2 
        super().__init__(n_particles=n_particles, dim=dim, **kwargs)




TARGETS_DICT = {
    "toy_sin_cos": ToySinCosEnergy,
    "double_well": DoubleWellEnergy,
    "lennard_jones": LennardJonesPotential
}