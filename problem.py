"""Drug diffusion with reaction in a 1D tissue slab.

Theta_t = Theta_xx - phi2 * Theta on 0 < x < 1, starting from zero concentration.
Left face: -Theta_x = bi * (1 - Theta). Right face: no flux.
The two unknowns are phi2 (reaction to diffusion ratio) and bi (surface mass transfer).

Both methods fit this linear model. The data generator can also use saturable clearance,
phi2 * Theta / (1 + kappa * Theta), to test what happens when the fitted model is wrong.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

T_END = 1.0
N_SENSORS = 50
N_TIMES = 110          # 50 x 110 gives 5500 observations per dataset
N_HELD = 500           # held out for model selection, the other 5000 are the fitting pool
REF_CELLS = 240        # mesh that makes the data
FIT_CELLS = 80         # coarser mesh inside the classical solver, avoids the inverse crime

REGIMES = {
    "moderate": (1.0, 1.0),
    "reaction": (25.0, 5.0),
    "permeability": (1.0, 0.1),
}

# bounds on log(phi2) and log(bi), and three starting guesses shared by both methods
LOW = np.log([0.01, 0.01])
HIGH = np.log([200.0, 50.0])
STARTS = [(0.2, 0.2), (3.0, 1.0), (10.0, 4.0)]


def diffusion(bi, cells):
    """Finite volume diffusion operator with the surface condition folded in."""
    h = 1.0 / cells
    g = 1.0 / (1.0 / bi + h / 2.0)  # surface and half cell resistances in series
    main = np.full(cells, -2.0 / h**2)
    main[0] = -1.0 / h**2 - g / h
    main[-1] = -1.0 / h**2
    off = np.full(cells - 1, 1.0 / h**2)
    b = np.zeros(cells)
    b[0] = g / h
    return main, off, b


def concentration(phi2, bi, x, t, cells, kappa=0.0):
    """Theta at positions x and times t, as an array of shape (len(t), len(x)).

    With kappa = 0 the semi discrete system is linear, theta' = A theta + b, and time is
    solved exactly with an eigendecomposition. With kappa > 0 a stiff solver is used.
    """
    h = 1.0 / cells
    main, off, b = diffusion(bi, cells)
    if kappa == 0.0:
        A = np.diag(main - phi2) + np.diag(off, 1) + np.diag(off, -1)
        steady = -np.linalg.solve(A, b)
        lam, V = np.linalg.eigh(A)
        coef = V.T @ steady
        inside = steady - (np.exp(np.outer(t, lam)) * coef) @ V.T  # starts from zero
    else:
        D = diags([off, main, off], [-1, 0, 1], format="csr")
        pattern = diags([np.ones(cells - 1), np.ones(cells), np.ones(cells - 1)], [-1, 0, 1])

        def rhs(_, th):
            return D @ th + b - phi2 * th / (1.0 + kappa * th)

        sol = solve_ivp(rhs, (0.0, t[-1]), np.zeros(cells), method="BDF", t_eval=t,
                        rtol=1e-8, atol=1e-10, jac_sparsity=pattern)
        inside = sol.y.T

    # add the two face values so sensors at x = 0 and x = 1 are interpolated, not clamped
    left = (bi + 2.0 / h * inside[:, 0]) / (bi + 2.0 / h)
    nodes = np.concatenate([[0.0], (np.arange(cells) + 0.5) * h, [1.0]])
    vals = np.column_stack([left, inside, inside[:, -1]])
    i = np.clip(np.searchsorted(nodes, x) - 1, 0, cells)
    w = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    return vals[:, i] * (1 - w) + vals[:, i + 1] * w


@dataclass
class Data:
    xs: np.ndarray      # sensor positions
    ts: np.ndarray      # observation times
    x: np.ndarray       # position of every observation, flattened
    t: np.ndarray       # time of every observation, flattened
    clean: np.ndarray
    noisy: np.ndarray
    fit: np.ndarray     # mask of observations used for fitting
    held: np.ndarray    # mask of held out observations
    sigma: float        # noise standard deviation
    truth: tuple

    def model(self, phi2, bi, cells=FIT_CELLS):
        return concentration(phi2, bi, self.xs, self.ts, cells).ravel()


def make_data(truth, noise, seed, n_fit=5000, kappa=0.0):
    """Fixed true parameters, accurate solution on the fine mesh, Gaussian noise, random split."""
    phi2, bi = truth
    xs = np.linspace(0.0, 1.0, N_SENSORS)
    ts = T_END * (np.arange(1, N_TIMES + 1) / N_TIMES) ** 2  # dense at early times
    clean = concentration(phi2, bi, xs, ts, REF_CELLS, kappa).ravel()

    rng = np.random.default_rng(seed)
    sigma = noise * clean.max()
    noisy = clean + sigma * rng.standard_normal(clean.size)

    order = rng.permutation(clean.size)
    held = np.zeros(clean.size, dtype=bool)
    fit = np.zeros(clean.size, dtype=bool)
    held[order[:N_HELD]] = True
    fit[order[N_HELD:N_HELD + n_fit]] = True

    xx, tt = np.meshgrid(xs, ts)
    return Data(xs, ts, xx.ravel(), tt.ravel(), clean, noisy, fit, held, sigma, (phi2, bi))


def info_bound(d, cells=FIT_CELLS):
    """Smallest possible relative error of any unbiased estimate, and the correlation between
    the two estimates. Returns (std of both relative errors as fractions, correlation).

    Cramer Rao bound for the two log parameters, from the sensitivity of the
    predictions to each parameter and the noise level of this dataset.
    Only valid when the fitted model is the true model. A correlation near 1 or minus 1
    means the two parameters trade off against each other in the data.
    """
    p = np.log(d.truth)
    eps = 1e-5
    cols = []
    for k in range(2):
        step = np.zeros(2)
        step[k] = eps
        up = d.model(*np.exp(p + step), cells)
        down = d.model(*np.exp(p - step), cells)
        cols.append((up - down) / (2 * eps))
    J = np.column_stack(cols)[d.fit]
    cov = d.sigma**2 * np.linalg.inv(J.T @ J)
    std = np.sqrt(np.diag(cov))
    return std, cov[0, 1] / (std[0] * std[1])
