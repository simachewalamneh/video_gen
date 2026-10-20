"""Forward noising, eps-prediction loss, DDPM + DDIM reverse sampling, linear/cosine schedules."""
import math, torch


def linear_betas(T):
    s = 1000 / T
    return torch.linspace(1e-4 * s, 0.02 * s, T, dtype=torch.float64)


def cosine_betas(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999)


class Diffusion:
    def __init__(self, T=200, schedule="cosine", device="cpu"):
        b = {"linear": linear_betas, "cosine": cosine_betas}[schedule](T)
        a = 1 - b
        ab = torch.cumprod(a, 0)
        abp = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
        d = lambda v: v.float().to(device)  # schedule cached on the right device
        self.T, self.schedule = T, schedule
        self.betas, self.ab, self.abp = d(b), d(ab), d(abp)
        self.sab, self.s1mab = d(ab.sqrt()), d((1 - ab).sqrt())
        self.pvar = d(b * (1 - abp) / (1 - ab))
        self.c1, self.c2 = d(b * abp.sqrt() / (1 - ab)), d((1 - abp) * a.sqrt() / (1 - ab))

    @staticmethod
    def _g(v, t):
        return v[t].view(-1, 1, 1, 1, 1)

    def q_sample(self, x0, t, eps):  # x_t = sqrt(ab) x0 + sqrt(1-ab) eps
        return self._g(self.sab, t) * x0 + self._g(self.s1mab, t) * eps

    def loss(self, model, x0, t=None):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device) if t is None else t
        eps = torch.randn_like(x0)
        return (model(self.q_sample(x0, t, eps), t) - eps).square().mean()

    def _x0(self, model, x, t):
        eps = model(x, t)
        return ((x - self._g(self.s1mab, t) * eps) / self._g(self.sab, t)).clamp(-1, 1), eps

    @torch.no_grad()
    def sample(self, model, shape, method="ddim", steps=50, device="cpu", seed=0):
        g = torch.Generator(device).manual_seed(seed)
        x = torch.randn(shape, device=device, generator=g)
        if method == "ddpm":
            for i in reversed(range(self.T)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x0, _ = self._x0(model, x, t)
                mean = self._g(self.c1, t) * x0 + self._g(self.c2, t) * x
                x = mean + (self._g(self.pvar, t).sqrt() * torch.randn(shape, device=device, generator=g) if i > 0 else 0)
            return x.clamp(-1, 1)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long().unique(sorted=True).flip(0)
        for k, i in enumerate(ts):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            x0, eps = self._x0(model, x, t)
            abp = self.ab[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=device)
            x = abp.sqrt() * x0 + (1 - abp).sqrt() * eps  # DDIM, eta = 0
        return x.clamp(-1, 1)
