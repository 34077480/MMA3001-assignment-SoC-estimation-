"""n-RC equivalent circuit model (ECM) with SoC- and temperature-dependent parameters.

Model
-----
::

    V      = OCV(s, T) + R0 i + sum_k v_k          (i < 0 = discharge, so v_k < 0 while discharging)
    dv_k/dt = -v_k / tau_k + (R_k / tau_k) i       k = 1..n_rc

Discretisation
--------------
Exact zero-order hold (ZOH): the current is constant over each sample period, so the linear ODE can be
integrated exactly over one step::

    v_k[n+1] = a_k v_k[n] + R_k (1 - a_k) i[n],    a_k = exp(-dt / tau_k)

This has no truncation error for piecewise-constant input and is unconditionally stable (0 < a_k < 1 for
any dt > 0), unlike forward Euler (a = 1 - dt/tau, unstable for dt > 2 tau). Because the update is exact,
fitted parameters remain valid at any sampling rate, which the sampling-rate sensitivity study relies on.
Verified against the analytical step response in ``tests/test_integrator.py``.

Parameterisation
----------------
Each parameter p in (R0, R1, tau1, R2, tau2, ...) is modelled in log space so it stays positive::

    log p(s, T) = g_p(s) + b1_p x + b2_p x^2,     x = 1000/(T + 273.15) - 1000/298.15

g_p is piecewise linear on ``KNOTS`` (denser at low SoC, where resistance rises steeply; clamped outside
[0, 1]); b1 is an Arrhenius slope and b2 lets the cold end bend. T is the *measured cell* temperature,
which matters because the cell self-heats by up to 17 degC during cold drive cycles.

``theta`` is an array of shape (1 + 2 n_rc, N_COEF): one row per parameter in the order R0, R1, tau1,
R2, tau2, ...; columns are the knot values of g_p followed by b1 and b2.

The kernels are compiled with numba so the fitter and the EKF run as tight scalar loops, which is also
how they would run on a microcontroller.
"""
import numpy as np
from numba import njit

KNOTS = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
"""SoC knots of the piecewise-linear parameter shapes."""
N_KNOTS = len(KNOTS)
N_COEF = N_KNOTS + 2
"""Coefficients per parameter: knot values + Arrhenius b1 + curvature b2."""
TREF_X = 1000.0 / 298.15


@njit(cache=True, inline="always")
def _param(c, s, x):
    """Value of one parameter and its derivative with respect to SoC.

    Parameters
    ----------
    c : numpy.ndarray
        One row of ``theta`` (length ``N_COEF``).
    s : float
        SoC (fraction); clamped to the knot range for the value.
    x : float
        Arrhenius variable ``1000/(T + 273.15) - 1000/298.15``.

    Returns
    -------
    (float, float)
        p and dp/ds (zero outside the knot range, where the value is clamped).
    """
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
    """OCV and its exact SoC derivative from the table of :class:`socest.ocv.OCV`.

    The derivative is the chord slope of the grid cell, i.e. the exact derivative of the linear
    interpolant, so the EKF Jacobian is consistent with the function it linearises (an earlier version
    used a separately smoothed slope and failed the finite-difference check).

    Parameters
    ----------
    tab : numpy.ndarray
        ``OCV.table``, shape (3, n).
    s0, ds : float
        First grid point and grid spacing.
    s : float
        SoC (fraction).
    T : float
        Cell temperature [degC].

    Returns
    -------
    (float, float)
        OCV [V] and dOCV/ds [V per unit SoC]. Outside the grid the value is clamped and the slope is 0.
    """
    n = tab.shape[1]
    f = (s - s0) / ds
    j = min(max(int(np.floor(f)), 0), n - 2)
    w = min(max(f - j, 0.0), 1.0)
    dT = T - 25.0
    val = 0.0
    der = 0.0
    p = 1.0
    for k in range(3):   # rows ocv25, k1, k2 are multiplied by 1, dT, dT^2
        val += (tab[k, j] * (1.0 - w) + tab[k, j + 1] * w) * p
        der += (tab[k, j + 1] - tab[k, j]) / ds * p
        p *= dT
    if f < 0.0 or f > n - 1:
        der = 0.0
    return val, der


@njit(cache=True)
def simulate(theta, s, T, i, dt, tab, s0, ds):
    """Simulate the terminal voltage for given SoC, temperature and current traces.

    SoC is an *input* here (e.g. the lab ground truth), which is how the model is fitted: the voltage
    residual then isolates the circuit model from any SoC-tracking error.

    Parameters
    ----------
    theta : numpy.ndarray
        Parameters, shape (1 + 2 n_rc, N_COEF).
    s, T, i : numpy.ndarray
        SoC (fraction), cell temperature [degC] and current [A, discharge negative], same length.
    dt : float
        Sample period [s].
    tab, s0, ds
        OCV table, see :func:`_ocv`.

    Returns
    -------
    numpy.ndarray
        Terminal voltage [V]. RC voltages start at zero (rested cell).
    """
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


def n_rc_of(theta):
    """Number of RC pairs encoded in ``theta``."""
    return (theta.shape[0] - 1) // 2


def params_at(theta, s, T):
    """Evaluate every parameter at one operating point, for reporting.

    Parameters
    ----------
    theta : numpy.ndarray
    s : float
        SoC (fraction).
    T : float
        Cell temperature [degC].

    Returns
    -------
    dict
        ``{"R0": ohm, "R1": ohm, "tau1": s, ...}``.
    """
    x = 1000.0 / (T + 273.15) - TREF_X
    names = ["R0"] + [f"{p}{m+1}" for m in range(n_rc_of(theta)) for p in ("R", "tau")]
    return {nm: float(_param(theta[r], s, x)[0]) for r, nm in enumerate(names)}


def initial_theta(n_rc):
    """Starting guess for the fitter: typical 18650 values at 25 degC, RC pairs ordered fast to slow.

    Parameters
    ----------
    n_rc : int
        1, 2 or 3.

    Returns
    -------
    numpy.ndarray
        Shape (1 + 2 n_rc, N_COEF).
    """
    if n_rc not in (1, 2, 3):
        raise ValueError("n_rc must be 1, 2 or 3")
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
