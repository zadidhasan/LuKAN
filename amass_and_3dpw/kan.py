import torch
from torch import nn
from einops.layers.torch import Rearrange
from KANS.Lucas import LucasPolynomials as LucasKAN

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_KAN(nn.Module):
    def __init__(self, dim):
        super(Spatial_KAN, self).__init__()
        self.kan = LucasKAN(dim, dim, 3)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.kan(x)
        x = self.arr1(x)
        return x

class Temporal_KAN(nn.Module):
    def __init__(self, dim):
        super(Temporal_KAN, self).__init__()
        self.kan = LucasKAN(dim, dim, 3)

    def forward(self, x):
        x = self.kan(x)
        return x



class KANblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:
            self.kan = Temporal_KAN(seq)
        else:
            self.kan = Spatial_KAN(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(180)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()


    def forward(self, x):
        x_ = self.kan(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x

class ApplyKANs(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.kans = nn.Sequential(*[
            KANblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.kans(x)
        return x

def kan_layers(args):
    return ApplyKANs(
        dim=args.hidden_dim,
        seq=args.dwt_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )

