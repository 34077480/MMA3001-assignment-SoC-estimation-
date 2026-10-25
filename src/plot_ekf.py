"""Trajectory figures: truth vs Coulomb counting vs EKF (with 3-sigma band) for selected runs/scenarios."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import DT, ROOT, load_cycle
from ekf import coulomb_count, ekf
from ocv import OCV
from run_ekf import P0_S
from sensors import SCENARIOS, corrupt

n_rc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = ROOT / "results"
ocv = OCV()
theta = np.load(OUT / f"ecm_theta_{n_rc}rc.npy")
q = json.loads((OUT / f"ekf_{n_rc}rc_eval.json").read_text())["q"]

cases = [(25, "UDDS", "clean"), (25, "UDDS", "soc0_-20%"), (25, "UDDS", "i_bias_+50mA"),
         (-10, "US06", "realistic"), (10, "HWFET", "realistic"), (-20, "UDDS", "realistic")]
fig, axes = plt.subplots(len(cases), 2, figsize=(13, 2.6 * len(cases)), gridspec_kw=dict(width_ratios=[2, 1]))
for (T, c, scn), (a0, a1) in zip(cases, axes):
    run = load_cycle(T, c)
    v, i, Tm, soc0 = corrupt(run, SCENARIOS[scn], 0)
    cc = coulomb_count(i, DT, soc0)
    soc, sig, _ = ekf(theta, v, i, Tm, DT, ocv.table, ocv.s0, ocv.ds, soc0, P0_S, *q)
    h = run.t / 3600
    a0.plot(h, run.soc * 100, "k", lw=1.5, label="truth (lab Ah)")
    a0.plot(h, cc * 100, lw=1, label="Coulomb counting")
    a0.plot(h, soc * 100, lw=1, label=f"EKF {n_rc}RC")
    a0.set(ylabel="SoC [%]", title=f"{T}C {c}, scenario: {scn}")
    a1.plot(h, (cc - run.soc) * 100, lw=0.8, label="CC error")
    a1.plot(h, (soc - run.soc) * 100, lw=0.8, label="EKF error")
    a1.fill_between(h, -300 * sig, 300 * sig, color="C1", alpha=0.2, lw=0, label="EKF ±3σ")
    a1.axhline(0, color="k", lw=0.5)
    a1.set(ylabel="error [% SoC]", ylim=(-12, 12))
    for a in (a0, a1):
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="best")
axes[-1, 0].set_xlabel("time [h]")
axes[-1, 1].set_xlabel("time [h]")
fig.tight_layout()
fig.savefig(OUT / "figures" / f"ekf_{n_rc}rc_trajectories.png", dpi=110)
print("saved")
