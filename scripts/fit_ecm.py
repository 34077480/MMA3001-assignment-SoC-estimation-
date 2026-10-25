"""Fit the n-RC ECM parameters by nonlinear least squares on the training drive cycles.

Residual: measured minus simulated terminal voltage, with the lab SoC as input, over all training cycles
at all training temperatures jointly (so the Arrhenius terms are identified across temperatures).

Solver: ``scipy.optimize.least_squares`` (trust-region reflective) with bounds on the log-parameters and a
2-point finite-difference Jacobian. Convergence: stops when the relative cost reduction falls below
ftol = 1e-8 (status 2) or after 200 evaluations.

Memory: the default 'exact' trust-region sub-solver takes a dense SVD of the (residuals x parameters)
Jacobian and holds several copies of it; for the 3-RC model at 5 Hz that is 592 k x 63 (285 MB per copy)
and it ran out of memory on a 16 GB laptop. The 3-RC fit therefore uses the iterative 'lsmr' sub-solver,
which needs only Jacobian-vector products (memory ~ one copy of J). 1- and 2-RC use 'exact'.

Residual sampling: the simulation always runs at 10 Hz; the residual is evaluated every ``--decim``
samples. Default 2 (5 Hz). An earlier 1 Hz setting (``--decim 10``, forced by a memory limit at the time)
gave similar voltage RMSE but worse SoC accuracy in the cold and less physical parameters; the comparison
is kept in the results (``*_d10`` files).

    python scripts/fit_ecm.py [--decim N] n_rc [n_rc ...]
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares

from socest.data import CYCLES, DT, RESULTS, TEMP_DIRS, TRAIN_CYCLES, TRAIN_TEMPS, load_cycle
from socest.ecm import initial_theta, params_at, simulate
from socest.ocv import OCV


def suffix(decim):
    """File-name suffix for a residual decimation (none for the default)."""
    return "" if decim == 2 else f"_d{decim}"


def fit(n_rc, runs, ocv, decim):
    """Fit one model; returns (theta, info dict)."""
    th0 = initial_theta(n_rc)
    shape = th0.shape
    tab, s0, ds = ocv.table, ocv.s0, ocv.ds

    def resid(p):
        """Voltage residual vector (simulated - measured) for a flat parameter vector."""
        th = p.reshape(shape)
        return np.concatenate([(simulate(th, r.soc, r.T, r.i, DT, tab, s0, ds) - r.v)[::decim] for r in runs])

    # bounds on log-parameters keep the optimiser physical and keep the RC pairs ordered fast -> slow
    lo, hi = np.full(shape, -np.inf), np.full(shape, np.inf)
    lo[:, -2:], hi[:, -2:] = -20, 20
    lo[0, :-2], hi[0, :-2] = np.log(1e-3), np.log(1.0)              # R0: 1 mOhm .. 1 Ohm
    for m in range(n_rc):
        lo[1 + 2 * m, :-2], hi[1 + 2 * m, :-2] = np.log(1e-4), np.log(1.0)
        tau_lo, tau_hi = [(0.1, 30), (30, 3000), (300, 10000)][m]
        lo[2 + 2 * m, :-2], hi[2 + 2 * m, :-2] = np.log(tau_lo), np.log(tau_hi)
    x0 = np.clip(th0, lo + 1e-9, hi - 1e-9).ravel()
    t0 = time.time()
    tr_solver = "lsmr" if n_rc >= 3 else "exact"
    sol = least_squares(resid, x0, bounds=(lo.ravel(), hi.ravel()), x_scale="jac", method="trf",
                        tr_solver=tr_solver, max_nfev=200, verbose=0)
    info = dict(n_rc=n_rc, decim=decim, tr_solver=tr_solver, nfev=int(sol.nfev), status=int(sol.status),
                time_s=time.time() - t0,
                train_rmse_mV=float(np.sqrt(np.mean(sol.fun ** 2)) * 1000), n_residuals=len(sol.fun),
                n_params=int(sol.x.size))
    print(f"  {n_rc}RC: {info['nfev']} evals, {info['time_s']:.0f}s, status {info['status']}, "
          f"train RMSE {info['train_rmse_mV']:.2f} mV", flush=True)
    return sol.x.reshape(shape), info


def evaluate(theta, ocv):
    """Voltage RMSE [mV] of a fitted model on every cycle at every temperature."""
    out = {}
    for T in TEMP_DIRS:
        for c in CYCLES:
            r = load_cycle(T, c)
            e = simulate(theta, r.soc, r.T, r.i, DT, ocv.table, ocv.s0, ocv.ds) - r.v
            out[f"{T}C_{c}"] = float(np.sqrt(np.mean(e ** 2)) * 1000)
    return out


def main():
    """Parse arguments, fit (or re-evaluate) the requested models and write reports."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n_rc", type=int, nargs="+")
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate saved parameters without refitting")
    a = ap.parse_args()
    ocv = OCV()
    ocv.save(RESULTS / "ocv_model.npz")      # used by socest.api without the raw data
    fitted = {}
    if a.eval_only:
        for n_rc in a.n_rc:
            fitted[n_rc] = (np.load(RESULTS / f"ecm_theta_{n_rc}rc{suffix(a.decim)}.npy"),
                            dict(n_rc=n_rc, decim=a.decim, note="re-evaluated; fit statistics not recorded"))
    else:
        runs = [load_cycle(T, c) for T in TRAIN_TEMPS for c in TRAIN_CYCLES]
        for n_rc in a.n_rc:
            fitted[n_rc] = fit(n_rc, runs, ocv, a.decim)
            np.save(RESULTS / f"ecm_theta_{n_rc}rc{suffix(a.decim)}.npy", fitted[n_rc][0])
        del runs
    for n_rc, (th, info) in fitted.items():
        rmse = evaluate(th, ocv)
        report = dict(info, rmse_mV=rmse, params_soc05={T: params_at(th, 0.5, T) for T in (25, 10, 0, -10, -20)})
        (RESULTS / f"ecm_fit_{n_rc}rc{suffix(a.decim)}.json").write_text(json.dumps(report, indent=1))
        print(f"{n_rc}RC voltage RMSE [mV], mean over 9 cycles (* = training temperature):")
        for T in TEMP_DIRS:
            vals = [rmse[f"{T}C_{c}"] for c in CYCLES]
            print(f"   {T:>4}C{'*' if T in TRAIN_TEMPS else ' '} {np.mean(vals):5.1f}")


if __name__ == "__main__":
    main()
