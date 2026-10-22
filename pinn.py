"""Inverse PINN: a network for Theta(x, s) plus two learnable physical parameters.

The network input is s = sqrt(t / T_END). Near the surface the solution changes like
sqrt(t) at early times, so in s the early layer is smooth and the observation grid,
which is dense at early times, becomes evenly spaced. The PDE in these coordinates is
u_s = 2 T s (u_xx - phi2 u).
"""
import time

import numpy as np
import torch
from torch import nn

from problem import T_END, LOW, HIGH
from settings import SMOKE

torch.set_default_dtype(torch.float64)

# untuned baseline. tune.py searches around it and saves the winner to results/pinn_config.json
DEFAULT = dict(layers=2, width=16, lr=3e-3, w_pde=1.0, w_data=3000.0, epochs=2000)
WARM = 0.2           # share of epochs that use the data term only
RAMP = 0.2           # share of epochs over which the physics terms are switched on
LBFGS_ITERS = 5 if SMOKE else 150
N_COLLOC = 2000      # fresh collocation points at every Adam step
N_BOUND = 200
W_BC = 1.0


class PINN(nn.Module):
    def __init__(self, layers, width, start):
        super().__init__()
        parts = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(layers - 1):
            parts += [nn.Linear(width, width), nn.Tanh()]
        parts.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*parts)
        self.lp = nn.Parameter(torch.tensor(np.log(start)))   # log(phi2), log(bi)
        self.register_buffer("low", torch.tensor(LOW))
        self.register_buffer("high", torch.tensor(HIGH))

    def params(self):
        return torch.exp(torch.clamp(self.lp, self.low, self.high))

    def u(self, x, s):
        # subtracting the network at s = 0 makes the initial condition exact
        return self.net(torch.cat([x, s], 1)) - self.net(torch.cat([x, torch.zeros_like(s)], 1))

    def loss(self, obs, colloc, sb, c, f):
        """Data term plus f times the physics terms. f ramps from 0 to 1 during training."""
        x, s, y = obs
        total = c["w_data"] * ((self.u(x, s) - y) ** 2).mean()
        if f == 0:
            return total
        phi2, bi = self.params()

        xc, sc = colloc
        u = self.u(xc, sc)
        ux = slope(u, xc)
        res = slope(u, sc) - 2 * T_END * sc * (slope(ux, xc) - phi2 * u)

        x0 = torch.zeros_like(sb).requires_grad_(True)
        u0 = self.u(x0, sb)
        robin = -slope(u0, x0) - bi * (1 - u0)
        x1 = torch.ones_like(sb).requires_grad_(True)
        flux = slope(self.u(x1, sb), x1)
        bc = (robin ** 2).mean() + (flux ** 2).mean()

        return total + f * (c["w_pde"] * (res ** 2).mean() + W_BC * bc)


def slope(out, var):
    return torch.autograd.grad(out.sum(), var, create_graph=True)[0]


def sample(n):
    return torch.rand(n, 1).requires_grad_(True), torch.rand(n, 1).requires_grad_(True)


def to_s(t):
    return torch.sqrt(torch.tensor(t) / T_END)[:, None]


def fit_pinn(d, start, seed, cfg=None):
    """Train one PINN. cfg overrides any key of DEFAULT."""
    c = {**DEFAULT, **(cfg or {})}
    epochs = min(c["epochs"], 60) if SMOKE else c["epochs"]
    torch.manual_seed(seed)
    model = PINN(c["layers"], c["width"], start)
    k = d.fit
    obs = (torch.tensor(d.x[k])[:, None], to_s(d.t[k]), torch.tensor(d.noisy[k])[:, None])

    t0 = time.perf_counter()
    opt = torch.optim.Adam([
        {"params": model.net.parameters(), "lr": c["lr"]},
        {"params": [model.lp], "lr": 2e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        done = ep / epochs
        f = 0.0 if done < WARM else min(1.0, (done - WARM) / RAMP)
        opt.zero_grad()
        model.loss(obs, sample(N_COLLOC), torch.rand(N_BOUND, 1), c, f).backward()
        opt.step()
        sched.step()

    colloc, sb = sample(N_COLLOC), torch.rand(N_BOUND, 1)
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = model.loss(obs, colloc, sb, c, 1.0)
        loss.backward()
        return loss

    lbfgs.step(closure)
    seconds = time.perf_counter() - t0

    lp = np.clip(model.lp.detach().numpy(), LOW, HIGH)
    with torch.no_grad():
        pred = model.u(torch.tensor(d.x)[:, None], to_s(d.t)).numpy().ravel()
    n_params = sum(p.numel() for p in model.net.parameters())
    phi2, bi = np.exp(lp)
    return dict(phi2=phi2, bi=bi, lp=lp, pred=pred, seconds=seconds, n_params=n_params)
