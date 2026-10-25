"""SoC error metrics and physical sanity checks.

All errors are reported in SoC percentage points (estimate - truth, times 100). Acceptance criterion used
throughout: RMSE <= 3 % (the commonly quoted automotive BMS target of 2-3 %).
"""
import numpy as np

from .data import DT, Q_REF

SETTLE_S = 600.0
"""'Settled' metrics skip the first 10 minutes, so an estimator's convergence from a wrong initial SoC
is not mixed up with its steady-state accuracy."""
TARGET_RMSE = 3.0
"""Acceptance threshold on RMSE [% SoC]."""


def soc_metrics(est, run, i_meas=None, sig=None):
    """Compute accuracy and sanity metrics for one SoC trajectory.

    Parameters
    ----------
    est : numpy.ndarray
        Estimated SoC (fraction) on the run's time grid.
    run : socest.data.Run
        Provides the ground truth ``run.soc`` and time ``run.t``.
    i_meas : numpy.ndarray, optional
        The current the estimator saw. Enables the charge-bookkeeping check.
    sig : numpy.ndarray, optional
        Reported SoC standard deviation (fraction). Enables the coverage check.

    Returns
    -------
    dict
        ``rmse``, ``maxabs``: over the whole run [% SoC];
        ``rmse_settled``, ``maxabs_settled``: after ``SETTLE_S``;
        ``final``: error at the end of the run;
        ``frac_outside_01``: fraction of estimates outside the physical range [0, 1];
        ``unexplained_dsoc`` (with ``i_meas``): change in the estimate not explained by the measured
        charge throughput [% SoC]. It is 0 for Coulomb counting by construction, equals the total Kalman
        correction for the EKF, and has no physical constraint at all for an ML model;
        ``coverage_3sig``, ``mean_sig`` (with ``sig``): fraction of settled samples whose error is within
        3 sigma, and the mean reported sigma [% SoC].
    """
    e = (est - run.soc) * 100
    late = run.t >= SETTLE_S
    out = dict(rmse=float(np.sqrt(np.mean(e ** 2))), maxabs=float(np.max(np.abs(e))),
               rmse_settled=float(np.sqrt(np.mean(e[late] ** 2))), maxabs_settled=float(np.max(np.abs(e[late]))),
               final=float(e[-1]),
               frac_outside_01=float(np.mean((est < 0) | (est > 1))))
    if i_meas is not None:
        cc = np.sum(i_meas[:-1]) * DT / (3600 * Q_REF)
        out["unexplained_dsoc"] = float(((est[-1] - est[0]) - cc) * 100)
    if sig is not None:
        out["coverage_3sig"] = float(np.mean(np.abs(e[late]) <= 300 * sig[late]))
        out["mean_sig"] = float(np.mean(sig[late]) * 100)
    return out
