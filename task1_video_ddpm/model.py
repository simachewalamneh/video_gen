"""Small 3D-conv denoiser with sinusoidal timestep embedding + residual blocks. Output shape == input shape."""
import math, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint


def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Res3D(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv3d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, e):
        h = self.c1(self.act(self.n1(x))) + self.t(e)[:, :, None, None, None]  # inject timestep
        h = self.c2(self.act(self.n2(h)))
        return h + self.skip(x)


class VideoDenoiser(nn.Module):
    def __init__(self, in_ch=1, ch=32, tdim=128, grad_ckpt=False):
        super().__init__()
        self.tdim, self.grad_ckpt = tdim, grad_ckpt
        self.temb = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim))
        self.inp = nn.Conv3d(in_ch, ch, 3, padding=1)
        self.d1 = Res3D(ch, ch, tdim)
        self.down = nn.Conv3d(ch, ch * 2, 3, stride=(1, 2, 2), padding=1)  # spatial downsample, keep time
        self.d2 = Res3D(ch * 2, ch * 2, tdim)
        self.mid = Res3D(ch * 2, ch * 2, tdim)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.u2 = Res3D(ch * 2 + ch, ch, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, in_ch, 3, padding=1))

    def _r(self, blk, x, e):
        return checkpoint(blk, x, e, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x, e)

    def forward(self, x, t):
        e = self.temb(timestep_embedding(t, self.tdim))
        h0 = self._r(self.d1, self.inp(x), e)
        h = self._r(self.d2, self.down(h0), e)
        h = self._r(self.mid, h, e)
        h = self._r(self.u2, torch.cat([self.up(h), h0], 1), e)
        return self.out(h)


if __name__ == "__main__":  # sanity checks: shape preservation + gradients
    m = VideoDenoiser()
    x, t = torch.randn(2, 1, 8, 32, 32), torch.randint(0, 200, (2,))
    y = m(x, t)
    assert y.shape == x.shape, "shape must be preserved"
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), "bad grads"
    print("OK | params: %.2fM | out %s" % (sum(p.numel() for p in m.parameters()) / 1e6, tuple(y.shape)))
