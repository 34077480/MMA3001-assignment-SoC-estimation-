"""Evaluate a trained recurrent SoC model on the same test set, scenarios and metrics as the EKF.

The 1 Hz output is mapped back to the 10 Hz grid causally (each estimate is held from the end of the
1-second block it was computed from), so metrics are computed on exactly the same samples as the EKF.
Sigma is calibrated on validation exactly as for the EKF.

    python src/run_ml.py <tag> [<tag> ...]      tags like lstm32_c500_s0; several tags = deep ensemble
"""
import json
import sys

import numpy as np
import torch

from data import ROOT, TEMP_DIRS, TEST_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle
from metrics import SETTLE_S, soc_metrics
from ml import FS_DECIM, SocNet, predict
from run_ekf import TUNE_SCENARIOS
from sensors import SCENARIOS, corrupt

OUT = ROOT / "results"


def load(tag):
    kind = "gru" if tag.startswith("gru") else "lstm"
    hidden = int(tag[len(kind):].split("_")[0])
    net = SocNet(kind, hidden)
    net.load_state_dict(torch.load(OUT / f"ml_{tag}.pt"))
    return net.eval()


def to_10hz(x, n):
    """Causal zero-order hold: value for block k becomes available at the last sample of block k."""
    y = np.repeat(x, FS_DECIM)
    y = np.concatenate([np.full(FS_DECIM - 1, y[0]), y])[:n]
    return np.concatenate([y, np.full(n - len(y), y[-1])]) if len(y) < n else y


def estimate(nets, run, sc, seed=0):
    v, i, T, _ = corrupt(run, sc, seed)       # the ML model takes no initial SoC
    outs = [predict(net, v, i, T) for net in nets]
    mu = np.mean([o[0] for o in outs], 0)
    # ensemble variance = mean aleatoric variance + spread of the means (epistemic)
    var = np.mean([o[1] ** 2 for o in outs], 0) + np.var([o[0] for o in outs], 0)
    n = len(run.t)
    return to_10hz(mu, n), to_10hz(np.sqrt(var), n), i


def main():
    tags = sys.argv[1:]
    nets = [load(t) for t in tags]
    name = tags[0] if len(tags) == 1 else f"ens{len(tags)}_{tags[0].rsplit('_s', 1)[0]}"
    val = [load_cycle(T, c) for T in TRAIN_TEMPS for c in VAL_CYCLES]
    ratios = []
    for r in val:
        for s in TUNE_SCENARIOS:
            mu, sig, _ = estimate(nets, r, SCENARIOS[s])
            late = r.t >= SETTLE_S
            ratios.append(np.abs(mu - r.soc)[late] / (3 * sig[late]))
    alpha = float(np.quantile(np.concatenate(ratios), 0.95))
    res = []
    for T in TEMP_DIRS:
        for c in TEST_CYCLES:
            run = load_cycle(T, c)
            for sname, sc in SCENARIOS.items():
                mu, sig, i = estimate(nets, run, sc)
                m = soc_metrics(mu, run, i, alpha * sig)
                m["coverage_3sig_raw"] = soc_metrics(mu, run, sig=sig)["coverage_3sig"]
                res.append(dict(temp=T, cycle=c, scenario=sname, unseen_temp=T not in TRAIN_TEMPS,
                                method=name, **m))
    (OUT / f"ml_{name}_eval.json").write_text(json.dumps(dict(alpha=alpha, tags=tags, results=res), indent=1))
    print(f"{name}: sigma alpha = {alpha:.2f}")
    print("  " + f"{'scenario':14s}" + "".join(f"{str(T)+('*' if T in TRAIN_TEMPS else ''):>8s}" for T in TEMP_DIRS))
    for sname in SCENARIOS:
        vals = [np.mean([r["rmse"] for r in res if r["scenario"] == sname and r["temp"] == T]) for T in TEMP_DIRS]
        print("  " + f"{sname:14s}" + "".join(f"{x:8.2f}" for x in vals))
    print(f"  settled max|err| (realistic) by T: " + " ".join(
        f"{T}:{np.max([r['maxabs_settled'] for r in res if r['scenario']=='realistic' and r['temp']==T]):.1f}"
        for T in TEMP_DIRS))
    print(f"  3-sigma coverage: raw {np.mean([r['coverage_3sig_raw'] for r in res])*100:.0f}%, "
          f"calibrated {np.mean([r['coverage_3sig'] for r in res])*100:.0f}%;  "
          f"fraction of outputs outside [0,1]: {np.mean([r['frac_outside_01'] for r in res])*100:.2f}%")


if __name__ == "__main__":
    main()
