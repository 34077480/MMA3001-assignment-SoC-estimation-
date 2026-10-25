"""Evaluate trained network(s) on the same test runs, scenarios and metrics as the EKF.

Sigma is calibrated on the validation cycle exactly as for the EKF. Several tags form a deep ensemble.

    python scripts/run_ml.py lstm32_c500_s0 [lstm32_c500_s1 ...] [--name NAME]
"""
import argparse
import json

import numpy as np

from socest.data import RESULTS, TEMP_DIRS, TEST_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle
from socest.metrics import SETTLE_S, soc_metrics
from socest.ml import estimate, load_model
from socest.sensors import SCENARIOS, TUNE_SCENARIOS, corrupt


def calibrate(nets):
    """Sigma scale alpha from the validation runs (95 % of settled errors within 3 alpha sigma)."""
    ratios = []
    for T in TRAIN_TEMPS:
        for c in VAL_CYCLES:
            r = load_cycle(T, c)
            for s in TUNE_SCENARIOS:
                v, i, Tm, _ = corrupt(r, SCENARIOS[s], 0)
                mu, sig = estimate(nets, v, i, Tm)
                late = r.t >= SETTLE_S
                ratios.append(np.abs(mu - r.soc)[late] / (3 * sig[late]))
    return float(np.quantile(np.concatenate(ratios), 0.95))


def evaluate(nets, alpha, name):
    """Metrics on every test run and scenario (the ML model ignores the initial-SoC guess)."""
    res = []
    for T in TEMP_DIRS:
        for c in TEST_CYCLES:
            run = load_cycle(T, c)
            for sname, sc in SCENARIOS.items():
                v, i, Tm, _ = corrupt(run, sc, 0)
                mu, sig = estimate(nets, v, i, Tm)
                m = soc_metrics(mu, run, i, alpha * sig)
                m["coverage_3sig_raw"] = soc_metrics(mu, run, sig=sig)["coverage_3sig"]
                res.append(dict(temp=T, cycle=c, scenario=sname, unseen_temp=T not in TRAIN_TEMPS,
                                method=name, **m))
    return res


def main():
    """Calibrate and evaluate the given networks (or ensemble) and save results."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    nets = [load_model(t) for t in a.tags]
    name = a.name or (a.tags[0] if len(a.tags) == 1 else f"ens{len(a.tags)}_{a.tags[0].rsplit('_s', 1)[0]}")
    alpha = calibrate(nets)
    res = evaluate(nets, alpha, name)
    (RESULTS / f"ml_{name}_eval.json").write_text(json.dumps(dict(alpha=alpha, tags=a.tags, results=res), indent=1))
    rr = [r for r in res if r["scenario"] == "realistic"]
    print(f"{name}: alpha {alpha:.2f}; realistic RMSE by T: " + " ".join(
        f"{T}:{np.mean([r['rmse'] for r in rr if r['temp'] == T]):.2f}" for T in TEMP_DIRS)
          + f"; coverage {np.mean([r['coverage_3sig'] for r in res])*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
