"""Tune the EKF on the validation cycle, then evaluate Coulomb counting and the EKF on the test cycles.

Tuning: grid search over (q_s, q_v, r) on LA92 at the training temperatures, averaged over several sensor
scenarios (tuning on clean lab data alone would teach the filter to trust Coulomb counting, which is only
safe because the lab sensor is perfect). We pick the lowest mean RMSE among settings whose 3-sigma band
contains the truth >= 90% of the time after settling, so the reported uncertainty means something. The
ECM's errors are slow (coloured) biases rather than white noise, so an RMSE-only tuning drives P towards
zero and the filter becomes confidently wrong.

    python src/run_ekf.py [n_rc] [suffix]    writes results/ekf_{n_rc}rc{suffix}_eval.json
    (suffix selects an alternative parameter file, e.g. _d2 for the fit with 5 Hz residuals)
"""
import itertools
import json
import sys
import time

import numpy as np

from data import DT, ROOT, TEMP_DIRS, TEST_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle
from ekf import coulomb_count, ekf
from metrics import SETTLE_S, soc_metrics
from ocv import OCV
from sensors import SCENARIOS, corrupt

OUT = ROOT / "results"
P0_S = 0.2 ** 2   # prior SoC variance: the filter is told it may be ~20 points off at start


def run_ekf(theta, ocv, run, sc, q, seed=0):
    v, i, T, soc0 = corrupt(run, sc, seed)
    soc, sig, innov = ekf(theta, v, i, T, DT, ocv.table, ocv.s0, ocv.ds, soc0, P0_S, *q)
    return soc, sig, innov, i


TUNE_SCENARIOS = ["clean", "soc0_-20%", "i_bias_+50mA", "realistic"]
MIN_COVERAGE = 0.9


def tune(theta, ocv):
    runs = [load_cycle(T, c) for T in TRAIN_TEMPS for c in VAL_CYCLES]
    grid = list(itertools.product([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],      # q_s (SoC^2 per step)
                                  [1e-9, 1e-8, 1e-7, 1e-6],                     # q_v (V^2 per step)
                                  [1.6e-3, 6.4e-3, 2.5e-2, 0.1, 0.4]))         # r (V^2): 40..630 mV
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
    print("  best 5 by RMSE (q_s, q_v, r):")
    for t in sorted(table, key=lambda t: t["rmse"])[:5]:
        print(f"    {t['q']}  RMSE {t['rmse']:.2f}%  3sig coverage {t['coverage']*100:.0f}%")
    print(f"  chosen (coverage >= {MIN_COVERAGE:.0%}, else best RMSE): {best['q']}  RMSE {best['rmse']:.2f}%  "
          f"coverage {best['coverage']*100:.0f}%")
    # Post-hoc calibration of the reported sigma (not of the filter gains): the smallest scale alpha such
    # that 95% of settled validation errors fall within 3 alpha sigma. The same procedure is applied to
    # the ML model's uncertainty, so the two are compared on equal terms.
    ratios = []
    for r in runs:
        for s in TUNE_SCENARIOS:
            soc, sig, _, _ = run_ekf(theta, ocv, r, SCENARIOS[s], best["q"])
            late = r.t >= SETTLE_S
            ratios.append(np.abs(soc - r.soc)[late] / (3 * sig[late]))
    alpha = float(np.quantile(np.concatenate(ratios), 0.95))
    print(f"  sigma calibration factor alpha = {alpha:.2f}")
    return best["q"], alpha, table


def main():
    n_rc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    ocv = OCV()
    theta = np.load(OUT / f"ecm_theta_{n_rc}rc{suffix}.npy")
    t0 = time.time()
    q, alpha, table = tune(theta, ocv)
    print(f"  tuned in {time.time()-t0:.0f}s -> q_s={q[0]:g}, q_v={q[1]:g}, r={q[2]:g}")
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
                res.append(dict(base, method=f"ekf_{n_rc}rc", **m,
                                innov_rms_mV=float(np.sqrt(np.mean(innov ** 2)) * 1000)))
    (OUT / f"ekf_{n_rc}rc{suffix}_eval.json").write_text(json.dumps(dict(q=q, alpha=alpha, tuning=table, results=res), indent=1))
    # compact console summary: RMSE (%) per temperature, averaged over test cycles
    print(f"\n  test-set RMSE [% SoC], mean over {TEST_CYCLES}  (* = training temperature)")
    print("  " + f"{'scenario':14s} {'method':9s}" + "".join(f"{str(T)+('*' if T in TRAIN_TEMPS else ''):>8s}" for T in TEMP_DIRS))
    for name in SCENARIOS:
        for m in ["coulomb", f"ekf_{n_rc}rc"]:
            vals = [np.mean([r["rmse"] for r in res if r["scenario"] == name and r["method"] == m and r["temp"] == T])
                    for T in TEMP_DIRS]
            print("  " + f"{name:14s} {m:9s}" + "".join(f"{x:8.2f}" for x in vals))
    ek = [r for r in res if r["method"].startswith("ekf")]
    print(f"\n  EKF test 3-sigma coverage: raw {np.mean([r['coverage_3sig_raw'] for r in ek])*100:.0f}%, "
          f"calibrated (alpha={alpha:.2f}) {np.mean([r['coverage_3sig'] for r in ek])*100:.0f}%")


if __name__ == "__main__":
    main()
