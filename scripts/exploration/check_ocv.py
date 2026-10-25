"""Validate the OCV model: C/20-derived curve vs 25C HPPC rests, and temperature model vs all HPPC rests."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from socest.data import ROOT
from socest.ocv import OCV, c20_curves, hppc_rest_points

ocv = OCV()
print("C/20-discharge correction [mV] at knots", np.round(ocv.tab["c20_offset"] * 1000, 1))
print("k1(s) [mV/degC] at knots  ", np.round(ocv.tab["k1"] * 1000, 2))
print("k2(s) [mV/degC^2] at knots", np.round(ocv.tab["k2"] * 1000, 4))
# finite-difference check of the tabulated derivative
s = np.linspace(0.05, 0.95, 50)
for T in (25, -10):
    h = 0.005  # a few grid cells, so this tests the tabulated slope rather than one cell's chord
    fd = (ocv(s + h, T) - ocv(s - h, T)) / (2 * h)
    print(f"dOCV/ds table vs finite diff at {T}C: max rel err {np.max(np.abs(ocv.dv_ds(s, T) - fd) / np.abs(fd)):.2e}")
print(f"{'T':>4} {'n':>3} {'RMSE mV':>8} {'max mV':>7} {'mean mV':>8}   (HPPC rest - model, 0.1<=SoC<=0.95)")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
(sd, vd), (sc, vc) = c20_curves()
ax[0].plot(sd, vd, lw=0.8, label="C/20 discharge")
ax[0].plot(sc, vc, lw=0.8, label="C/20 charge")
ax[0].plot(ocv.s, ocv(ocv.s, 25), "k", lw=1.2, label="OCV model 25C")
for T in [25, 10, 0, -10, -20]:
    p = hppc_rest_points(T)
    m = (p[:, 0] >= 0.1) & (p[:, 0] <= 0.95)
    e = (p[m, 1] - ocv(p[m, 0], T)) * 1000
    tag = "train" if T in (25, 0, -20) else "UNSEEN"
    print(f"{T:>4} {m.sum():>3} {np.sqrt(np.mean(e**2)):8.1f} {np.abs(e).max():7.1f} {e.mean():8.1f}   {tag}")
    l, = ax[1].plot(p[:, 0], p[:, 1] - ocv(p[:, 0], 25), "o", ms=3, label=f"HPPC rest {T}C")
    ax[1].plot(ocv.s, ocv(ocv.s, T) - ocv(ocv.s, 25), color=l.get_color(), lw=1)
ax[0].plot(*hppc_rest_points(25).T, "o", ms=3, mfc="none", label="HPPC rest 25C")
ax[0].set(xlabel="SoC", ylabel="V", title="OCV at 25C", xlim=(-0.05, 1.02))
ax[1].set(xlabel="SoC", ylabel="OCV(T) - OCV(25C)  [V]", title="Temperature dependence (lines = model)",
          xlim=(0, 1.02), ylim=(-0.08, 0.02))
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "ocv.png", dpi=130)
