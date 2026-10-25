"""Open-circuit voltage model OCV(SoC, T) built from the characterisation tests.

Construction
------------
* **25 degC shape and level.** The dense C/20 discharge curve gives the shape. A smooth piecewise-linear
  correction fitted to the relaxed HPPC rest voltages (34-64 min rests) sets the level. The C/20 *charge*
  curve is deliberately not used: it sits ~130 mV above the relaxed rests, so the common trick of
  averaging charge and discharge curves biases OCV upward by ~65 mV on this cell (found during
  development; see ``scripts/exploration/check_ocv.py``).
* **Smoothing.** The C/20 log has mV-level noise, which, differentiated on a 0.001 SoC grid, gives a
  useless dOCV/dSoC (the EKF's most important Jacobian entry). The curve is Savitzky-Golay smoothed over
  +-0.015 SoC and forced to be strictly increasing so the inverse exists.
* **Temperature.** ``OCV(s, T) = ocv25(s) + k1(s) dT + k2(s) dT^2`` with ``dT = T - 25``, fitted to HPPC rest
  voltages at the *training* temperatures only (25, 0, -20 degC). The quadratic term is needed because the
  -20 degC rests are not fully relaxed after 30 min, so the apparent OCV falls faster at the cold end.

Validation against the HPPC rests (``scripts/exploration/check_ocv.py``): RMSE 4.5 mV at 25 degC, 6-8 mV
at the training temperatures and 7-8 mV at the unseen 10 and -10 degC.

The model is tabulated on a uniform SoC grid so the numba kernels can evaluate it with one index
computation (see ``socest.ecm._ocv``).
"""
import numpy as np
from scipy.signal import savgol_filter

from .data import Q_REF, RAW, TRAIN_TEMPS, load_mat

S_GRID = np.linspace(-0.05, 1.05, 1101)
"""SoC grid of the table: 0.001 resolution with a 5 % margin on both sides for estimator excursions."""
K_KNOTS = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
"""SoC knots of the piecewise-linear correction and temperature coefficients."""


def c20_curves():
    """Return the 25 degC C/20 discharge and charge curves.

    Returns
    -------
    (tuple of numpy.ndarray, tuple of numpy.ndarray)
        ``(s_dis, v_dis), (s_chg, v_chg)``: SoC (fraction) and terminal voltage [V] of each curve.
    """
    d = load_mat(next(RAW.rglob("*C20 OCV Test_C20_25dC.mat")))
    i, v, ah = d["i"], d["v"], d["ah"]
    dis = np.flatnonzero(i < -0.1)
    chg = np.flatnonzero(i > 0.1)
    s = 1 + (ah - ah[dis[0]]) / Q_REF
    return (s[dis], v[dis]), (s[chg], v[chg])


def hppc_rest_points(temp):
    """Relaxed-voltage points from the HPPC test at one temperature.

    A point is taken at the end of every rest of at least 30 minutes that precedes a pulse group.

    Parameters
    ----------
    temp : int
        Ambient temperature [degC]: 25, 10, 0, -10 or -20.

    Returns
    -------
    numpy.ndarray
        Shape (n, 2): columns SoC (fraction) and voltage [V], ordered from full to empty.

    Raises
    ------
    FileNotFoundError
        If exactly one HPPC file for ``temp`` cannot be found.
    """
    # names are "<date>_<time> <descriptor>.mat": anchor on the space so "0degC" can't match "n10degC"
    name = f" {'n' if temp < 0 else ''}{abs(temp)}degC_5".lower()
    hits = [p for p in RAW.rglob("*HPPC*.mat") if name in p.name.lower()]
    if len(hits) != 1:
        raise FileNotFoundError(f"HPPC file for {temp} degC: found {hits}")
    d = load_mat(hits[0])
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
    """Piecewise-linear 'hat' basis on ``knots`` evaluated at ``s`` (clamped to the knot range).

    Returns an array of shape (len(s), len(knots)) whose rows sum to one.
    """
    s = np.clip(np.atleast_1d(s), knots[0], knots[-1])
    B = np.zeros((len(s), len(knots)))
    j = np.clip(np.searchsorted(knots, s) - 1, 0, len(knots) - 2)
    w = (s - knots[j]) / (knots[j + 1] - knots[j])
    B[np.arange(len(s)), j] = 1 - w
    B[np.arange(len(s)), j + 1] = w
    return B


def build():
    """Fit the OCV model from the raw characterisation data.

    Returns
    -------
    dict
        ``s`` (grid), ``ocv25`` (V on the grid), ``k_knots``, ``k1`` [V/degC] and ``k2`` [V/degC^2] at the
        knots, and ``c20_offset`` (the level correction applied to the C/20 curve, V at the knots).
    """
    (sd, vd), _ = c20_curves()
    od = np.argsort(sd)
    vd_g = np.interp(S_GRID, sd[od], vd[od])
    p25 = hppc_rest_points(25)
    B = _hat(p25[:, 0], K_KNOTS)
    D = np.diff(np.eye(len(K_KNOTS)), axis=0)
    off = np.linalg.solve(B.T @ B + 1e-2 * D.T @ D, B.T @ (p25[:, 1] - np.interp(p25[:, 0], S_GRID, vd_g)))
    ocv25 = vd_g + _hat(S_GRID, K_KNOTS) @ off
    # below the end of the C/20 discharge, extrapolate linearly with the slope of the last 1 % SoC
    lo = S_GRID < sd.min()
    if lo.any():
        s0 = sd.min()
        slope = (np.interp(s0 + 0.01, S_GRID, ocv25) - np.interp(s0, S_GRID, ocv25)) / 0.01
        ocv25[lo] = np.interp(s0, S_GRID, ocv25) + slope * (S_GRID[lo] - s0)
    ocv25 = savgol_filter(ocv25, 31, 3, mode="interp")
    ocv25 = np.maximum.accumulate(ocv25 + 1e-6 * np.arange(len(S_GRID)))

    # Temperature terms, referenced to the 25 degC HPPC rests (not to the C/20 curve) so that any
    # C/20-vs-HPPC offset cancels. Only the cold *training* temperatures are used.
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
    D = np.kron(np.diag([1.0, 30.0]), np.diff(np.eye(n), axis=0))  # smoothness along SoC
    lam = 1e-3  # small ridge: knots with no cold data (low SoC at -20 degC) borrow from neighbours
    k = np.linalg.solve(A.T @ A + lam * D.T @ D * len(y), A.T @ y)
    return dict(s=S_GRID, ocv25=ocv25, k_knots=K_KNOTS, k1=k[:n], k2=k[n:], c20_offset=off)


class OCV:
    """Tabulated open-circuit voltage ``OCV(s, T) = ocv25(s) + k1(s) dT + k2(s) dT^2``, ``dT = T - 25``.

    Parameters
    ----------
    tab : dict, optional
        Output of :func:`build`. Built from the raw data if omitted.

    Attributes
    ----------
    table : numpy.ndarray
        Shape (3, n): rows ``ocv25``, ``k1`` and ``k2`` on the uniform grid ``s``. Passed directly to the
        numba kernels together with ``s0`` (first grid point) and ``ds`` (grid spacing).
    """

    def __init__(self, tab=None):
        tab = tab or build()
        self.tab = tab
        self.s = tab["s"]
        self.s0, self.ds = float(self.s[0]), float(self.s[1] - self.s[0])
        H = _hat(self.s, tab["k_knots"])
        self.table = np.ascontiguousarray(np.vstack([tab["ocv25"], H @ tab["k1"], H @ tab["k2"]]))

    def _rows(self, s, T, rows):
        dT = np.asarray(T, float) - 25
        return rows[0](s) + rows[1](s) * dT + rows[2](s) * dT ** 2

    def __call__(self, s, T):
        """OCV [V] at SoC ``s`` (fraction) and temperature ``T`` [degC]; array inputs broadcast."""
        return self._rows(s, T, [lambda x, r=r: np.interp(x, self.s, r) for r in self.table])

    def dv_ds(self, s, T):
        """dOCV/dSoC [V per unit SoC]: the exact slope of the linear interpolant, as used by the EKF."""
        slopes = np.diff(self.table, axis=1) / self.ds
        idx = lambda x: np.clip(((np.asarray(x) - self.s0) / self.ds).astype(int), 0, len(self.s) - 2)
        return self._rows(s, T, [lambda x, r=r: r[idx(x)] for r in slopes])

    def save(self, path):
        """Save the fitted model (the ``build`` output) to an ``.npz`` file."""
        np.savez(path, **{k: np.asarray(v) for k, v in self.tab.items()})

    @classmethod
    def load(cls, path):
        """Load a model saved with :meth:`save`; needs no raw data."""
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def inverse(self, v, T):
        """SoC whose OCV equals ``v`` at temperature ``T`` (scalar), clipped to [0, 1].

        Used to initialise an estimator from a rested voltage, as a BMS does at power-up. Only valid if
        the cell has rested long enough for the RC voltages to decay (tens of minutes; longer in the cold).
        """
        curve = self(self.s, T)
        return float(np.clip(np.interp(v, curve, self.s), 0.0, 1.0))
