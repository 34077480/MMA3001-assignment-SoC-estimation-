"""SoC error metrics and physical sanity checks. All errors in SoC percentage points."""
import numpy as np

from data import DT, Q_REF

SETTLE_S = 600.0   # 'after convergence' metrics skip the first 10 minutes


def soc_metrics(est, run, i_meas=None, sig=None):
    e = (est - run.soc) * 100
    late = run.t >= SETTLE_S
    out = dict(rmse=float(np.sqrt(np.mean(e ** 2))), maxabs=float(np.max(np.abs(e))),
               rmse_settled=float(np.sqrt(np.mean(e[late] ** 2))), maxabs_settled=float(np.max(np.abs(e[late]))),
               final=float(e[-1]),
               frac_outside_01=float(np.mean((est < 0) | (est > 1))))
    if i_meas is not None:
        # charge bookkeeping: how much of the estimate's SoC change is NOT explained by the measured
        # current (for Coulomb counting this is 0 by construction; for the EKF it is the total Kalman
        # correction; for an ML model it is whatever it does)
        cc = np.sum(i_meas[:-1]) * DT / (3600 * Q_REF)
        out["unexplained_dsoc"] = float(((est[-1] - est[0]) - cc) * 100)
    if sig is not None:
        out["coverage_3sig"] = float(np.mean(np.abs(e[late]) <= 300 * sig[late]))
        out["mean_sig"] = float(np.mean(sig[late]) * 100)
    return out
