"""Fit ECM parameters by nonlinear least squares on the training drive cycles (train temperatures only).

Residual = measured terminal voltage - simulated voltage, with the lab SoC as input. Fitted jointly over
all training runs so the Arrhenius terms are identified across temperatures. Usage:
    python src/fit_ecm.py [n_rc ...]      (default 1 2)
Set ECM_DECIM=2 to compare residuals at 5 Hz instead of 1 Hz; results are then saved with a "_d2" suffix.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

from data import DT, ROOT, TEMP_DIRS, TRAIN_CYCLES, TRAIN_TEMPS, CYCLES, load_cycle
from ecm import N_COEF, initial_theta, params_at, simulate
from ocv import OCV

DECIM = int(os.environ.get("ECM_DECIM", 10))  # 10 = residual at 1 Hz (simulation still runs at 10 Hz)
SUFFIX = "" if DECIM == 10 else f"_d{DECIM}"
OUT = ROOT / "results"


def fit(n_rc, runs, ocv):
    th0 = initial_theta(n_rc)
    shape = th0.shape
    tab, s0, ds = ocv.table, ocv.s0, ocv.ds

    def resid(p):
        th = p.reshape(shape)
        return np.concatenate([(simulate(th, r.soc, r.T, r.i, DT, tab, s0, ds) - r.v)[::DECIM] for r in runs])

    # bound log-parameters to physically sensible ranges so the optimiser can't wander off
    lo, hi = np.full(shape, -np.inf), np.full(shape, np.inf)
    lo[:, -2:], hi[:, -2:] = -20, 20
    lo[0, :-2], hi[0, :-2] = np.log(1e-3), np.log(1.0)            # R0 1 mOhm - 1 Ohm
    for m in range(n_rc):
        lo[1 + 2 * m, :-2], hi[1 + 2 * m, :-2] = np.log(1e-4), np.log(1.0)
        tau_lo, tau_hi = [(0.1, 30), (30, 3000), (300, 10000)][m]  # keep pairs ordered fast -> slow
        lo[2 + 2 * m, :-2], hi[2 + 2 * m, :-2] = np.log(tau_lo), np.log(tau_hi)
    x0 = np.clip(th0, lo + 1e-9, hi - 1e-9).ravel()
    t0 = time.time()
    sol = least_squares(resid, x0, bounds=(lo.ravel(), hi.ravel()), x_scale="jac", method="trf",
                        max_nfev=200, verbose=0)
    print(f"  {n_rc}RC: {sol.nfev} evals, {time.time()-t0:.0f}s, status {sol.status}, "
          f"train RMSE {np.sqrt(np.mean(sol.fun**2))*1000:.2f} mV")
    return sol.x.reshape(shape)


def main():
    ocv = OCV()
    runs = [load_cycle(T, c) for T in TRAIN_TEMPS for c in TRAIN_CYCLES]
    thetas = {}
    for n_rc in [int(a) for a in sys.argv[1:]] or [1, 2]:
        thetas[n_rc] = fit(n_rc, runs, ocv)
        np.save(OUT / f"ecm_theta_{n_rc}rc{SUFFIX}.npy", thetas[n_rc])
    del runs
    results = {}
    for n_rc, th in thetas.items():
        print(f"{n_rc}RC")
        rmse = {}
        for T in TEMP_DIRS:
            for c in CYCLES:
                r = load_cycle(T, c)
                e = simulate(th, r.soc, r.T, r.i, DT, ocv.table, ocv.s0, ocv.ds) - r.v
                rmse[f"{T}C_{c}"] = float(np.sqrt(np.mean(e ** 2)) * 1000)
        results[n_rc] = rmse
        print(f"  voltage RMSE [mV] by temperature (all 9 cycles; * = training temperature):")
        for T in TEMP_DIRS:
            vals = [rmse[f"{T}C_{c}"] for c in CYCLES]
            print(f"   {T:>4}C{'*' if T in TRAIN_TEMPS else ' '} mean {np.mean(vals):5.1f}  "
                  + " ".join(f"{c}:{v:.1f}" for c, v in zip(CYCLES, vals)))
        for T in (25, 0, -20):
            print(f"   params at SoC 0.5, {T}C: "
                  + ", ".join(f"{k}={v*1000:.1f}m" if k.startswith("R") else f"{k}={v:.1f}s"
                              for k, v in params_at(th, 0.5, T).items()))
    (OUT / f"ecm_fit_rmse_mV{SUFFIX}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
