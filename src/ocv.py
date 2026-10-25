"""Open-circuit voltage OCV(SoC, T).

* 25C: the dense C/20 discharge curve gives the shape; a smooth piecewise-linear correction fitted to the
  relaxed HPPC rest voltages (34-64 min rests) at 25C sets the level. The C/20 charge curve is not used:
  it sits ~130 mV above the relaxed rests (hysteresis/charge-side overpotential), so averaging it in, the
  usual trick, biases OCV upward by ~65 mV on this cell.
* Temperature: HPPC rest voltages (>=30 min rest before each pulse group) at the TRAINING temperatures give
  OCV_T(s) - OCV_25(s); we fit dOCV = k(s) * (T - 25) with k piecewise linear in SoC. 10C and -10C are not
  used, so they stay unseen.

The result is tabulated on a fine SoC grid so OCV and dOCV/dSoC are cheap lookups inside the EKF.
"""
import numpy as np
from scipy.signal import savgol_filter

from data import Q_REF, RAW, TRAIN_TEMPS, load_mat

S_GRID = np.linspace(-0.05, 1.05, 1101)   # 0.001 SoC resolution, with margin for estimator excursions
K_KNOTS = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])


def c20_curves():
    d = load_mat(next(RAW.rglob("*C20 OCV Test_C20_25dC.mat")))
    i, v, ah = d["i"], d["v"], d["ah"]
    dis = np.flatnonzero(i < -0.1)
    chg = np.flatnonzero(i > 0.1)
    ah0 = ah[dis[0]]
    s = 1 + (ah - ah0) / Q_REF
    return (s[dis], v[dis]), (s[chg], v[chg])


def hppc_rest_points(temp):
    """(SoC, V) at the end of each long rest (>=30 min) before an HPPC pulse group."""
    # file names are "<date>_<time> <descriptor>.mat"; anchor on the space so "0degC" can't match "n10degC"
    name = f" {'n' if temp < 0 else ''}{abs(temp)}degC_5".lower()
    hits = [p for p in RAW.rglob("*HPPC*.mat") if name in p.name.lower()]
    assert len(hits) == 1, (temp, hits)
    f = hits[0]
    d = load_mat(f)
    t, i, v, ah = d["t"], d["i"], d["v"], d["ah"]
    active = np.abs(i) > 0.02
    starts = np.flatnonzero(np.diff(active.astype(int)) == 1) + 1
    ends = np.flatnonzero(np.diff(active.astype(int)) == -1) + 1
    pts = []
    for s in starts:
        prev = ends[ends < s]
        rest = t[s - 1] - t[prev[-1]] if len(prev) else np.inf
        if rest >= 1800:
            pts.append((1 + ah[s - 1] / Q_REF, v[s - 1]))
    return np.array(pts)


def _hat(s, knots):
    """Piecewise-linear 'hat' basis on knots, evaluated at s (clamped)."""
    s = np.clip(np.atleast_1d(s), knots[0], knots[-1])
    B = np.zeros((len(s), len(knots)))
    j = np.clip(np.searchsorted(knots, s) - 1, 0, len(knots) - 2)
    w = (s - knots[j]) / (knots[j + 1] - knots[j])
    B[np.arange(len(s)), j] = 1 - w
    B[np.arange(len(s)), j + 1] = w
    return B


def build():
    (sd, vd), _ = c20_curves()
    od = np.argsort(sd)
    vd_g = np.interp(S_GRID, sd[od], vd[od])
    p25 = hppc_rest_points(25)
    B = _hat(p25[:, 0], K_KNOTS)
    D = np.diff(np.eye(len(K_KNOTS)), axis=0)
    off = np.linalg.solve(B.T @ B + 1e-2 * D.T @ D, B.T @ (p25[:, 1] - np.interp(p25[:, 0], S_GRID, vd_g)))
    ocv25 = vd_g + _hat(S_GRID, K_KNOTS) @ off
    # below the end of the C/20 discharge, extrapolate linearly using the last 1% of slope
    lo = S_GRID < sd.min()
    if lo.any():
        s0 = sd.min()
        slope = (np.interp(s0 + 0.01, S_GRID, ocv25) - np.interp(s0, S_GRID, ocv25)) / 0.01
        ocv25[lo] = np.interp(s0, S_GRID, ocv25) + slope * (S_GRID[lo] - s0)
    # The C/20 log has mV-level noise, which differentiated on a 0.001 grid gives a useless dOCV/dSoC
    # (the EKF's main Jacobian entry). Smooth over +-0.015 SoC, then enforce monotonicity.
    ocv25 = savgol_filter(ocv25, 31, 3, mode="interp")
    ocv25 = np.maximum.accumulate(ocv25 + 1e-6 * np.arange(len(S_GRID)))

    # temperature coefficient k(s) [V/degC] from HPPC rests at the cold training temperatures,
    # referenced to the 25C HPPC rests (not the C/20 curve) so any C/20-vs-HPPC offset cancels
    # dOCV = k1(s) dT + k2(s) dT^2, dT = T - 25. Quadratic because the -20C rests are not fully relaxed
    # after the >=30 min rest, so the apparent OCV drops faster at the cold end; with three training
    # temperatures the quadratic is exactly determined at each SoC.
    rows, y = [], []
    for T in TRAIN_TEMPS:
        if T == 25:
            continue
        p = hppc_rest_points(T)
        d = p[:, 1] - np.interp(p[:, 0], p25[::-1, 0], p25[::-1, 1])
        B = _hat(p[:, 0], K_KNOTS)
        rows.append(np.hstack([B * (T - 25), B * (T - 25) ** 2]))
        y.append(d)
    A, y = np.vstack(rows), np.concatenate(y)
    n = len(K_KNOTS)
    D = np.kron(np.diag([1.0, 30.0]), np.diff(np.eye(n), axis=0))  # smoothness along SoC, k2 scaled to k1
    lam = 1e-3  # small ridge so knots with no cold data (low SoC at -20C) fall back to neighbours
    k = np.linalg.solve(A.T @ A + lam * D.T @ D * len(y), A.T @ y)
    return dict(s=S_GRID, ocv25=ocv25, k_knots=K_KNOTS, k1=k[:n], k2=k[n:], c20_offset=off)


class OCV:
    """OCV(s, T) = ocv25(s) + k1(s) dT + k2(s) dT^2 with dT = T - 25, tabulated on a uniform SoC grid.

    `table` is a (6, n) array [ocv25, k1, k2, d/ds of each] that the numba kernels index directly."""

    def __init__(self, tab=None):
        tab = tab or build()
        self.tab = tab
        self.s = tab["s"]
        self.s0, self.ds = self.s[0], self.s[1] - self.s[0]
        H = _hat(self.s, tab["k_knots"])
        rows = [tab["ocv25"], H @ tab["k1"], H @ tab["k2"]]
        self.table = np.vstack(rows + [np.gradient(r, self.ds) for r in rows])

    def _eval(self, s, T, off):
        dT = np.asarray(T, float) - 25
        f = lambda r: np.interp(s, self.s, self.table[off + r])
        return f(0) + f(1) * dT + f(2) * dT ** 2

    def __call__(self, s, T):
        return self._eval(s, T, 0)

    def dv_ds(self, s, T):
        return self._eval(s, T, 3)
