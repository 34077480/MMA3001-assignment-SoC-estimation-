"""Checking and cleaning of estimator inputs (invalid, missing and unsupported data).

Policy
------
| Condition | Handling |
|---|---|
| Arrays of different length, not 1-D, or fewer than 2 samples | `ValueError` |
| Non-numeric / complex data | `ValueError` |
| Values outside physical hard limits (e.g. V > 5 V, T > 100 degC) | `ValueError` (sensor fault or wrong units; the message suggests the likely unit mistake, e.g. mV or kelvin) |
| Missing samples (NaN/inf), gaps up to `max_gap_s` | Linearly interpolated, `InputDomainWarning` issued |
| Missing samples, longer gaps | `ValueError`: the estimator would be running blind |
| Non-uniform or non-increasing time stamps (if `t` is given) | `ValueError`: resample to a uniform grid first |
| Inside the hard limits but outside the range seen in training (e.g. T < -21 or > 34 degC, I < -21.5 A) | `InputDomainWarning`: the models extrapolate there and accuracy is not validated |

Warnings rather than errors for out-of-domain data is deliberate: a BMS must keep producing an estimate,
but the caller should know the validation evidence does not cover it.
"""
import warnings

import numpy as np


class InputDomainWarning(UserWarning):
    """Input was repaired, or lies outside the range the models were trained and validated on."""


HARD_LIMITS = dict(v=(0.0, 5.0), i=(-100.0, 100.0), T=(-60.0, 100.0))
"""Physically implausible for a single 18650 cell: treated as sensor faults."""
VALIDATED = dict(v=(2.4, 4.25), i=(-21.5, 11.0), T=(-21.0, 34.0))
"""Ranges covered by the training and test data (cell-level V [V], I [A], case T [degC])."""
_UNIT_HINT = dict(v="voltage in mV instead of V?", i="current in mA instead of A?",
                  T="temperature in kelvin instead of degC?")


def _fill_gaps(x, name, max_gap):
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.all():
        raise ValueError(f"{name}: no finite samples")
    # length of the longest run of consecutive missing samples
    edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    longest = int(np.max(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)))
    if longest > max_gap:
        raise ValueError(f"{name}: {longest} consecutive missing samples exceeds the {max_gap}-sample limit")
    idx = np.arange(len(x))
    warnings.warn(f"{name}: interpolated {int(bad.sum())} missing samples (longest gap {longest})",
                  InputDomainWarning, stacklevel=3)
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out


def validate_inputs(v, i, T, dt=0.1, t=None, max_gap_s=1.0):
    """Validate and clean one record of measurements.

    Parameters
    ----------
    v : array_like
        Terminal voltage [V].
    i : array_like
        Current [A], discharge negative.
    T : array_like
        Cell temperature [degC].
    dt : float, optional
        Nominal sample period [s].
    t : array_like, optional
        Time stamps [s]. If given, they must be strictly increasing and uniform to within 1 %.
    max_gap_s : float, optional
        Longest run of missing samples [s] that may be interpolated.

    Returns
    -------
    v, i, T : numpy.ndarray
        Cleaned float64 copies.

    Raises
    ------
    ValueError
        See the module policy table.

    Warns
    -----
    InputDomainWarning
        Missing samples were interpolated, or values lie outside the validated range.
    """
    if not (dt > 0):
        raise ValueError("dt must be positive")
    arrs = {}
    for name, x in (("v", v), ("i", i), ("T", T)):
        a = np.asarray(x)
        if a.dtype.kind not in "iuf":
            raise ValueError(f"{name}: expected real numbers, got dtype {a.dtype}")
        a = a.astype(float)
        if a.ndim != 1:
            raise ValueError(f"{name}: expected a 1-D array, got shape {a.shape}")
        arrs[name] = a
    n = len(arrs["v"])
    if any(len(a) != n for a in arrs.values()):
        raise ValueError(f"v, i, T must have equal length, got {[len(a) for a in arrs.values()]}")
    if n < 2:
        raise ValueError("need at least 2 samples")
    if t is not None:
        t = np.asarray(t, float)
        if len(t) != n:
            raise ValueError("t must have the same length as v")
        d = np.diff(t)
        if np.any(d <= 0):
            raise ValueError("time stamps must be strictly increasing")
        if np.max(np.abs(d - dt)) > 0.01 * dt:
            raise ValueError(f"non-uniform sampling (dt from {d.min():.4g} to {d.max():.4g} s); "
                             "resample onto a uniform grid first")
    max_gap = max(1, int(round(max_gap_s / dt)))
    for name in arrs:
        a = _fill_gaps(arrs[name], name, max_gap)
        lo, hi = HARD_LIMITS[name]
        if a.min() < lo or a.max() > hi:
            raise ValueError(f"{name} outside physical limits [{lo}, {hi}] "
                             f"(range {a.min():.4g}..{a.max():.4g}); {_UNIT_HINT[name]}")
        lo, hi = VALIDATED[name]
        frac = float(np.mean((a < lo) | (a > hi)))
        if frac > 0:
            warnings.warn(f"{name}: {frac*100:.1f}% of samples outside the validated range [{lo}, {hi}]; "
                          "the estimate is an extrapolation there", InputDomainWarning, stacklevel=2)
        arrs[name] = a
    return arrs["v"], arrs["i"], arrs["T"]
