"""n-RC equivalent circuit model with SoC- and temperature-dependent parameters.

    V = OCV(s, T) + R0 i + sum_k v_k            (i < 0 = discharge, so v_k < 0 while discharging)
    dv_k/dt = -v_k / tau_k + (R_k / tau_k) i

Discretisation: exact zero-order hold (current constant over each sample period),
    v_k[n+1] = a_k v_k[n] + R_k (1 - a_k) i[n],   a_k = exp(-dt / tau_k)
which is exact for piecewise-constant input and unconditionally stable for any dt > 0 (0 < a_k < 1),
unlike forward Euler (a = 1 - dt/tau, unstable for dt > 2 tau).

Each parameter p in (R0, R1, tau1, R2, tau2, ...) is modelled in log space:
    log p(s, T) = g_p(s) + b1_p x + b2_p x^2,   x = 1000/(T + 273.15) - 1000/298.15
g_p is piecewise linear on SoC knots (denser at low SoC, where resistance rises steeply; clamped outside),
b1 is an Arrhenius slope and b2 lets the cold end bend. Log space keeps every parameter positive.
T is the measured cell temperature.
"""
import numpy as np
from numba import njit

KNOTS = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
N_KNOTS = len(KNOTS)
N_COEF = N_KNOTS + 2
TREF_X = 1000.0 / 298.15


@njit(cache=True, inline="always")
def _param(c, s, x):
    """Value of one parameter and its derivative wrt SoC. c = [knot coefs..., b1, b2]."""
    sc = min(max(s, KNOTS[0]), KNOTS[-1])
    j = 0
    while j < N_KNOTS - 2 and sc > KNOTS[j + 1]:
        j += 1
    h = KNOTS[j + 1] - KNOTS[j]
    w = (sc - KNOTS[j]) / h
    g = c[j] * (1.0 - w) + c[j + 1] * w
    dg = (c[j + 1] - c[j]) / h if KNOTS[0] < s < KNOTS[-1] else 0.0
    p = np.exp(g + c[N_KNOTS] * x + c[N_KNOTS + 1] * x * x)
    return p, p * dg


@njit(cache=True, inline="always")
def _ocv(tab, s0, ds, s, T):
    """OCV by linear interpolation of the table from ocv.OCV, and its exact derivative (the cell's chord
    slope), so the EKF Jacobian is consistent with the function it linearises."""
    n = tab.shape[1]
    f = (s - s0) / ds
    j = min(max(int(np.floor(f)), 0), n - 2)
    w = min(max(f - j, 0.0), 1.0)
    dT = T - 25.0
    val = 0.0
    der = 0.0
    p = 1.0
    for k in range(3):   # rows 0..2: ocv25, k1, k2 multiply 1, dT, dT^2
        val += (tab[k, j] * (1.0 - w) + tab[k, j + 1] * w) * p
        der += (tab[k, j + 1] - tab[k, j]) / ds * p
        p *= dT
    if f < 0.0 or f > n - 1:
        der = 0.0        # clamped outside the table
    return val, der


@njit(cache=True)
def simulate(theta, s, T, i, dt, tab, s0, ds):
    """Terminal voltage for given SoC/temperature/current traces (SoC supplied, e.g. ground truth).

    theta has shape (1 + 2 n_rc, N_COEF): rows R0, R1, tau1, R2, tau2, ..."""
    n = len(i)
    n_rc = (theta.shape[0] - 1) // 2
    v = np.empty(n)
    vrc = np.zeros(n_rc)
    for k in range(n):
        x = 1000.0 / (T[k] + 273.15) - TREF_X
        ocv, _ = _ocv(tab, s0, ds, s[k], T[k])
        r0, _ = _param(theta[0], s[k], x)
        acc = ocv + r0 * i[k]
        for m in range(n_rc):
            acc += vrc[m]
        v[k] = acc
        for m in range(n_rc):
            R, _ = _param(theta[1 + 2 * m], s[k], x)
            tau, _ = _param(theta[2 + 2 * m], s[k], x)
            a = np.exp(-dt / tau)
            vrc[m] = a * vrc[m] + R * (1.0 - a) * i[k]
    return v


def params_at(theta, s, T):
    """Evaluate all parameters at scalar (s, T) -> dict, for reporting."""
    x = 1000.0 / (T + 273.15) - TREF_X
    names = ["R0"] + [f"{p}{m+1}" for m in range((theta.shape[0] - 1) // 2) for p in ("R", "tau")]
    return {nm: _param(theta[r], s, x)[0] for r, nm in enumerate(names)}


def initial_theta(n_rc):
    """Rough 18650 values at 25C: R0 25 mOhm, then RC pairs from fast to slow."""
    guess = [(0.025, 2.0)] + [(0.010, 3.0), (0.015, 200.0), (0.010, 1500.0)][:n_rc]
    th = np.zeros((1 + 2 * n_rc, N_COEF))
    th[0, :N_KNOTS] = np.log(guess[0][0])
    th[0, N_KNOTS] = 2.0   # resistances grow in the cold
    for m in range(n_rc):
        R, tau = guess[1 + m]
        th[1 + 2 * m, :N_KNOTS] = np.log(R)
        th[1 + 2 * m, N_KNOTS] = 2.0
        th[2 + 2 * m, :N_KNOTS] = np.log(tau)
    return th
