"""Mean voltage residual (sim - meas) by SoC bin, per temperature and cycle, for a fitted ECM."""
import sys

import numpy as np

from socest.data import DT, ROOT, TEMP_DIRS, load_cycle
from socest.ecm import simulate
from socest.ocv import OCV

n_rc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ocv = OCV()
th = np.load(ROOT / "results" / f"ecm_theta_{n_rc}rc.npy")
bins = np.arange(0, 1.01, 0.1)
print("mean(sim - meas) [mV] by SoC bin " + " ".join(f"{b:5.1f}" for b in bins[:-1]) + "   | rest-only (|I|<0.1A)")
for T in TEMP_DIRS:
    for c in ["Cycle_1", "LA92", "UDDS", "US06"]:
        r = load_cycle(T, c)
        e = (simulate(th, r.soc, r.T, r.i, DT, ocv.table, ocv.s0, ocv.ds) - r.v) * 1000
        idx = np.digitize(r.soc, bins) - 1
        rest = np.abs(r.i) < 0.1
        m = [np.mean(e[idx == k]) if np.sum(idx == k) > 100 else np.nan for k in range(10)]
        mr = [np.mean(e[(idx == k) & rest]) if np.sum((idx == k) & rest) > 100 else np.nan for k in range(10)]
        print(f"{T:>4}C {c:8s} " + " ".join(f"{x:5.0f}" for x in m) + "   | " + " ".join(f"{x:4.0f}" for x in mr))
