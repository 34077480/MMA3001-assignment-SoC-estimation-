"""Coulomb counting (baseline) and an Extended Kalman Filter on the n-RC ECM.

State ``x = [SoC, v_1, ..., v_n]``. Per sample k, with measured (V_k, I_k, T_k)::

  update   y_k = OCV(s, T) + R0(s, T) I_k + sum v_m
           H   = [dOCV/ds + dR0/ds I_k, 1, ..., 1]
           K   = P H^T / (H P H^T + r);  x += K (V_k - y_k)
           P   = (I - K H) P (I - K H)^T + r K K^T          (Joseph form: stays symmetric PSD)
  predict  s   <- s + I_k dt / (3600 Q)
           v_m <- a_m v_m + R_m (1 - a_m) I_k,   a_m = exp(-dt / tau_m(s, T))
           F   = d(predict)/dx: F[0,0] = 1, F[m,m] = a_m,
                 F[m,0] = (da_m/ds) v_m + (dR_m/ds (1 - a_m) - R_m da_m/ds) I_k,
                 da_m/ds = a_m dt / tau_m^2 * dtau_m/ds
           P   <- F P F^T + Q

The estimate reported at step k is the posterior after the update with measurement k.

Numerical-stability choices
---------------------------
* Joseph-form covariance update: keeps P symmetric positive semi-definite under rounding, which the
  short form ``P = (I - K H) P`` does not guarantee.
* Scalar measurement: the innovation variance S is a scalar, so no matrix inversion is needed.
* The ZOH transition factor ``a_m`` is always in (0, 1), so the prediction cannot diverge.
* The Jacobians are analytic and checked against central finite differences in ``tests/test_ekf.py``.
"""
import numpy as np
from numba import njit

from .data import Q_REF
from .ecm import TREF_X, _ocv, _param


@njit(cache=True)
def coulomb_count(i, dt, soc0, q_ah=Q_REF):
    """Coulomb counting: integrate current from an initial SoC (forward Euler on dSoC/dt = i / 3600 Q).

    For a piecewise-constant current this rectangle rule is exact. It has no feedback, so an error in
    ``soc0`` persists unchanged and a current-sensor bias b drifts the estimate by b t / (3600 Q).

    Parameters
    ----------
    i : numpy.ndarray
        Current [A], discharge negative.
    dt : float
        Sample period [s].
    soc0 : float
        Initial SoC (fraction).
    q_ah : float, optional
        Capacity [Ah] used for normalisation.

    Returns
    -------
    numpy.ndarray
        SoC estimate (fraction), same length as ``i``.
    """
    n = len(i)
    s = np.empty(n)
    s[0] = soc0
    for k in range(n - 1):
        s[k + 1] = s[k] + i[k] * dt / (3600.0 * q_ah)
    return s


@njit(cache=True)
def ekf(theta, v, i, T, dt, tab, s0_tab, ds_tab, soc0, p0_s, q_s, q_v, r):
    """Run the EKF over a full record.

    Parameters
    ----------
    theta : numpy.ndarray
        ECM parameters, shape (1 + 2 n_rc, N_COEF); see :mod:`socest.ecm`.
    v, i, T : numpy.ndarray
        Measured voltage [V], current [A, discharge negative] and cell temperature [degC].
    dt : float
        Sample period [s].
    tab, s0_tab, ds_tab
        OCV table (``OCV.table``, ``OCV.s0``, ``OCV.ds``).
    soc0 : float
        Initial SoC guess (fraction).
    p0_s : float
        Initial SoC variance (fraction^2), i.e. how uncertain ``soc0`` is.
    q_s : float
        Process-noise variance of SoC per step (fraction^2).
    q_v : float
        Process-noise variance of each RC voltage per step (V^2).
    r : float
        Measurement-noise variance (V^2). In practice it must also cover the ECM's own voltage error.

    Returns
    -------
    soc : numpy.ndarray
        Posterior SoC estimate (fraction).
    sigma : numpy.ndarray
        Posterior SoC standard deviation, sqrt(P[0, 0]).
    innov : numpy.ndarray
        Innovation (measured minus predicted voltage) [V].
    """
    n = len(v)
    n_rc = (theta.shape[0] - 1) // 2
    nx = 1 + n_rc
    x = np.zeros(nx)
    x[0] = soc0
    P = np.zeros((nx, nx))
    P[0, 0] = p0_s
    for m in range(n_rc):
        P[1 + m, 1 + m] = 1e-4                      # RC voltages start near zero (cell rested)
    Qd = np.zeros(nx)
    Qd[0] = q_s
    for m in range(n_rc):
        Qd[1 + m] = q_v
    soc = np.empty(n)
    sig = np.empty(n)
    innov = np.empty(n)
    H = np.zeros(nx)
    F = np.zeros((nx, nx))
    PHt = np.zeros(nx)
    K = np.zeros(nx)
    A = np.zeros((nx, nx))
    tmp = np.zeros((nx, nx))
    for k in range(n):
        xk = 1000.0 / (T[k] + 273.15) - TREF_X
        s = x[0]
        # ---- measurement update
        ocv, docv = _ocv(tab, s0_tab, ds_tab, s, T[k])
        r0, dr0 = _param(theta[0], s, xk)
        y = ocv + r0 * i[k]
        H[0] = docv + dr0 * i[k]
        for m in range(n_rc):
            y += x[1 + m]
            H[1 + m] = 1.0
        for a in range(nx):
            acc = 0.0
            for b in range(nx):
                acc += P[a, b] * H[b]
            PHt[a] = acc
        S = r
        for a in range(nx):
            S += H[a] * PHt[a]
        e = v[k] - y
        for a in range(nx):
            K[a] = PHt[a] / S
            x[a] += K[a] * e
        # Joseph form: P = A P A^T + r K K^T, A = I - K H
        for a in range(nx):
            for b in range(nx):
                A[a, b] = (1.0 if a == b else 0.0) - K[a] * H[b]
        for a in range(nx):
            for b in range(nx):
                acc = 0.0
                for c in range(nx):
                    acc += A[a, c] * P[c, b]
                tmp[a, b] = acc
        for a in range(nx):
            for b in range(nx):
                acc = 0.0
                for c in range(nx):
                    acc += tmp[a, c] * A[b, c]
                P[a, b] = acc + r * K[a] * K[b]
        soc[k] = x[0]
        sig[k] = np.sqrt(max(P[0, 0], 0.0))
        innov[k] = e
        # ---- time update (uses the posterior SoC for the parameter lookup)
        s = x[0]
        F[:, :] = 0.0
        F[0, 0] = 1.0
        x[0] = s + i[k] * dt / (3600.0 * Q_REF)
        for m in range(n_rc):
            R, dR = _param(theta[1 + 2 * m], s, xk)
            tau, dtau = _param(theta[2 + 2 * m], s, xk)
            a_ = np.exp(-dt / tau)
            da = a_ * dt / (tau * tau) * dtau
            vm = x[1 + m]
            F[1 + m, 1 + m] = a_
            F[1 + m, 0] = da * vm + (dR * (1.0 - a_) - R * da) * i[k]
            x[1 + m] = a_ * vm + R * (1.0 - a_) * i[k]
        for a in range(nx):
            for b in range(nx):
                acc = 0.0
                for c in range(nx):
                    acc += F[a, c] * P[c, b]
                tmp[a, b] = acc
        for a in range(nx):
            for b in range(nx):
                acc = 0.0
                for c in range(nx):
                    acc += tmp[a, c] * F[b, c]
                P[a, b] = acc + (Qd[a] if a == b else 0.0)
    return soc, sig, innov


def ekf_reference(theta, v, i, T, dt, tab, s0_tab, ds_tab, soc0, p0_s, q_s, q_v, r):
    """Readable NumPy implementation of the same filter, without numba.

    It uses the same parameter/OCV lookups (called from Python) and ordinary matrix algebra. Two purposes:
    an independent implementation that :func:`ekf` must reproduce to rounding error
    (``tests/test_ekf.py``), and the unoptimised baseline in the runtime comparison (``scripts/perf.py``).
    Same parameters and returns as :func:`ekf`.
    """
    n_rc = (theta.shape[0] - 1) // 2
    nx = 1 + n_rc
    x = np.zeros(nx)
    x[0] = soc0
    P = np.diag([p0_s] + [1e-4] * n_rc)
    Q = np.diag([q_s] + [q_v] * n_rc)
    I = np.eye(nx)
    out = np.empty((3, len(v)))
    for k in range(len(v)):
        xk = 1000.0 / (T[k] + 273.15) - TREF_X
        ocv, docv = _ocv(tab, s0_tab, ds_tab, x[0], T[k])
        r0, dr0 = _param(theta[0], x[0], xk)
        H = np.ones(nx)
        H[0] = docv + dr0 * i[k]
        e = v[k] - (ocv + r0 * i[k] + x[1:].sum())
        K = P @ H / (H @ P @ H + r)
        x = x + K * e
        A = I - np.outer(K, H)
        P = A @ P @ A.T + r * np.outer(K, K)
        out[:, k] = x[0], np.sqrt(max(P[0, 0], 0.0)), e
        s = x[0]
        F = np.eye(nx)
        xn = x.copy()
        xn[0] = s + i[k] * dt / (3600.0 * Q_REF)
        for m in range(n_rc):
            R, dR = _param(theta[1 + 2 * m], s, xk)
            tau, dtau = _param(theta[2 + 2 * m], s, xk)
            a = np.exp(-dt / tau)
            da = a * dt / tau ** 2 * dtau
            F[1 + m, 1 + m] = a
            F[1 + m, 0] = da * x[1 + m] + (dR * (1 - a) - R * da) * i[k]
            xn[1 + m] = a * x[1 + m] + R * (1 - a) * i[k]
        x = xn
        P = F @ P @ F.T + Q
    return out[0], out[1], out[2]
