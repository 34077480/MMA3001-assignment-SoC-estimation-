"""Coulomb counting and an Extended Kalman Filter on the n-RC ECM.

State x = [SoC, v_1, ..., v_n]. Per sample k, with measured (V_k, I_k, T_k):

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
"""
import numpy as np
from numba import njit

from data import Q_REF
from ecm import TREF_X, _ocv, _param


@njit(cache=True)
def coulomb_count(i, dt, soc0, q_ah=Q_REF):
    n = len(i)
    s = np.empty(n)
    s[0] = soc0
    for k in range(n - 1):
        s[k + 1] = s[k] + i[k] * dt / (3600.0 * q_ah)
    return s


@njit(cache=True)
def ekf(theta, v, i, T, dt, tab, s0_tab, ds_tab, soc0, p0_s, q_s, q_v, r):
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
