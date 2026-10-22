"""Classical inverse solver: nonlinear least squares on the PDE solution."""
import time

import numpy as np
from scipy.optimize import least_squares

from problem import LOW, HIGH


def fit_classical(d, start):
    """Adjust log(phi2) and log(bi) until the solver output matches the fitting observations."""
    def resid(lp):
        return d.model(*np.exp(lp))[d.fit] - d.noisy[d.fit]

    t0 = time.perf_counter()
    out = least_squares(resid, np.log(start), bounds=(LOW, HIGH), diff_step=1e-5)
    seconds = time.perf_counter() - t0

    phi2, bi = np.exp(out.x)
    return dict(phi2=phi2, bi=bi, lp=out.x, pred=d.model(phi2, bi), seconds=seconds)
