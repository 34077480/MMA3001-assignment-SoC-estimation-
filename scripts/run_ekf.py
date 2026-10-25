"""Tune the EKF on the validation cycle, then evaluate Coulomb counting and the EKF on the test cycles.

Tuning: grid search over (q_s, q_v, r) on LA92 at the training temperatures, averaged over the
``TUNE_SCENARIOS`` sensor scenarios. Tuning on clean lab data alone would teach the filter to trust
Coulomb counting, which is only safe because the lab sensor is perfect. The chosen setting is the lowest
mean RMSE among those whose 3-sigma band contains the truth >= 90 % of the time after settling. The ECM's
errors are slow (coloured) biases rather than white noise, so an RMSE-only tuning drives P towards zero
and the filter becomes confidently wrong (seen in development: sigma 0.05 % with 1-13 % actual error).

Uncertainty calibration: one scale factor alpha on sigma such that 95 % of settled validation errors fall
within 3 alpha sigma. The same procedure is applied to the ML model.

    python scripts/run_ekf.py [n_rc] [--suffix _d10]     writes results/ekf_{n_rc}rc{suffix}_eval.json
"""
import argparse
import itertools
import json
import time

import numpy as np

from socest.api import P0_S
from socest.data import DT, RESULTS, TEMP_DIRS, TEST_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle
from socest.ekf import coulomb_count, ekf
from socest.metrics import SETTLE_S, soc_metrics
from socest.ocv import OCV
from socest.sensors import SCENARIOS, TUNE_SCENARIOS, corrupt

MIN_COVERAGE = 0.9
GRID = list(itertools.product([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],    # q_s (SoC^2 per 0.1 s step)
                              [1e-9, 1e-8, 1e-7, 1e-6],                   # q_v (V^2 per step)
                              [1.6e-3, 6.4e-3, 2.5e-2, 0.1, 0.4]))       # r (V^2): 40..630 mV
"""Tuning grid (120 settings)."""


def run_ekf(theta, ocv, run, sc, q, seed=0, dt=DT):
    """Run the EKF on one run under one scenario. Returns (soc, sigma, innovation, measured current)."""
    v, i, T, soc0 = corrupt(run, sc, seed)
    soc, sig, innov = ekf(theta, v, i, T, dt, ocv.table, ocv.s0, ocv.ds, soc0, P0_S, *q)
    return soc, sig, innov, i


def tune(theta, ocv, runs=None, grid=GRID, verbose=True):
    """Grid-search (q_s, q_v, r) on the validation runs; returns (q, alpha, table)."""
    runs = runs or [load_cycle(T, c) for T in TRAIN_TEMPS for c in VAL_CYCLES]
    table = []
    for q in grid:
        ms = []
        for r in runs:
            for s in TUNE_SCENARIOS:
                soc, sig, _, _ = run_ekf(theta, ocv, r, SCENARIOS[s], q)
                ms.append(soc_metrics(soc, r, sig=sig))
        table.append(dict(q=q, rmse=float(np.mean([m["rmse"] for m in ms])),
                          coverage=float(np.mean([m["coverage_3sig"] for m in ms]))))
    ok = [t for t in table if t["coverage"] >= MIN_COVERAGE] or table
    best = min(ok, key=lambda t: t["rmse"])
    ratios = []
    for r in runs:
        for s in TUNE_SCENARIOS:
            soc, sig, _, _ = run_ekf(theta, ocv, r, SCENARIOS[s], best["q"])
            late = r.t >= SETTLE_S
            ratios.append(np.abs(soc - r.soc)[late] / (3 * sig[late]))
    alpha = float(np.quantile(np.concatenate(ratios), 0.95))
    if verbose:
        print("  best 5 by RMSE (q_s, q_v, r):")
        for t in sorted(table, key=lambda t: t["rmse"])[:5]:
            print(f"    {t['q']}  RMSE {t['rmse']:.2f}%  3sig coverage {t['coverage']*100:.0f}%")
        print(f"  chosen: {best['q']}  RMSE {best['rmse']:.2f}%  coverage {best['coverage']*100:.0f}%  "
              f"alpha {alpha:.2f}")
    return best["q"], alpha, table


def evaluate(theta, ocv, q, alpha, label):
    """Coulomb counting and EKF on every test run and scenario; returns a list of metric dicts."""
    res = []
    for T in TEMP_DIRS:
        for c in TEST_CYCLES:
            run = load_cycle(T, c)
            for name, sc in SCENARIOS.items():
                v, i, Tm, soc0 = corrupt(run, sc, 0)
                cc = coulomb_count(i, DT, soc0)
                soc, sig, innov, _ = run_ekf(theta, ocv, run, sc, q)
                base = dict(temp=T, cycle=c, scenario=name, unseen_temp=T not in TRAIN_TEMPS)
                res.append(dict(base, method="coulomb", **soc_metrics(cc, run, i)))
                m = soc_metrics(soc, run, i, alpha * sig)
                m["coverage_3sig_raw"] = soc_metrics(soc, run, sig=sig)["coverage_3sig"]
                res.append(dict(base, method=label, **m, innov_rms_mV=float(np.sqrt(np.mean(innov ** 2)) * 1000)))
    return res


def summary(res, methods):
    """Print test RMSE per scenario and temperature."""
    print(f"  test RMSE [% SoC], mean over {TEST_CYCLES}  (* = training temperature)")
    print("  " + f"{'scenario':14s} {'method':10s}" + "".join(f"{str(T)+('*' if T in TRAIN_TEMPS else ''):>8s}"
                                                         for T in TEMP_DIRS))
    for name in SCENARIOS:
        for m in methods:
            vals = [np.mean([r["rmse"] for r in res if r["scenario"] == name and r["method"] == m and r["temp"] == T])
                    for T in TEMP_DIRS]
            print("  " + f"{name:14s} {m:10s}" + "".join(f"{x:8.2f}" for x in vals))


def main():
    """Tune, evaluate and save results for one ECM order."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("n_rc", type=int, nargs="?", default=2)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    ocv = OCV()
    theta = np.load(RESULTS / f"ecm_theta_{a.n_rc}rc{a.suffix}.npy")
    t0 = time.time()
    q, alpha, table = tune(theta, ocv)
    print(f"  tuned in {time.time()-t0:.0f}s")
    label = f"ekf_{a.n_rc}rc"
    res = evaluate(theta, ocv, q, alpha, label)
    (RESULTS / f"ekf_{a.n_rc}rc{a.suffix}_eval.json").write_text(
        json.dumps(dict(q=q, alpha=alpha, tuning=table, results=res), indent=1))
    summary(res, ["coulomb", label])
    ek = [r for r in res if r["method"] == label]
    print(f"  EKF test 3-sigma coverage: raw {np.mean([r['coverage_3sig_raw'] for r in ek])*100:.0f}%, "
          f"calibrated {np.mean([r['coverage_3sig'] for r in ek])*100:.0f}%")


if __name__ == "__main__":
    main()
