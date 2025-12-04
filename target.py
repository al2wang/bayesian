import torch
class Target:
    pass


class ToySinCos(Target):
    def __init__(self, dim=1, domain=[-2.5, 2.5]):
        self.dim = dim
        self.name = f"toy_sin_cos"
        self.file_name = f"toy_sin_cos_dim{self.dim}"
        d0 = torch.tensor(domain[0])
        d1 = torch.tensor(domain[1])
        if len(d0.shape) == 0:
            d0 = torch.zeros(self.dim) + d0
        if len(d1.shape) == 0:
            d1 = torch.zeros(self.dim) + d1

        assert list(d0.shape) == [self.dim]
        assert list(d1.shape) == [self.dim]
        self.domain = [d0, d1]
    
    def __call__(self, x):  # x: (Batch, Dim)
        return torch.sum(torch.sin(x) * torch.cos(x), dim=1, keepdim=True)

    def sample_domain_uniform(self, num=1):
        return torch.rand(num, self.dim) * (self.domain[1] - self.domain[0]) + self.domain[0]
    
TARGETS_DICT = {
    "toy_sin_cos": ToySinCos
}