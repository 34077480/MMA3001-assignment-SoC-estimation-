import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from socest.data import DT, ROOT, load_cycle
from socest.ecm import simulate
from socest.ocv import OCV

ocv = OCV()
th = np.load(ROOT / "results" / "ecm_theta_1rc.npy")
fig, ax = plt.subplots(3, 2, figsize=(13, 9))
for col, (T, c) in enumerate([(25, "UDDS"), (0, "Cycle_1")]):
    r = load_cycle(T, c)
    vs = simulate(th, r.soc, r.T, r.i, DT, ocv.table, ocv.s0, ocv.ds)
    e = vs - r.v
    sl = slice(int(3000 / DT), int(3300 / DT))
    ax[0, col].plot(r.t[sl], r.v[sl], lw=0.8, label="measured")
    ax[0, col].plot(r.t[sl], vs[sl], lw=0.8, label="1RC sim")
    ax[0, col].plot(r.t[sl], ocv(r.soc[sl], r.T[sl]), lw=0.8, label="OCV(true SoC)")
    ax[0, col].set_title(f"{T}C {c}: 300 s zoom")
    ax[0, col].legend(fontsize=7)
    ax[1, col].plot(r.soc, e * 1000, lw=0.3)
    ax[1, col].set(xlabel="SoC", ylabel="sim - meas [mV]")
    # binned residual by SoC, and mean voltage minus OCV at near-zero current (rest-like samples)
    rest = np.abs(r.i) < 0.05
    bins = np.linspace(0, 1, 21)
    idx = np.digitize(r.soc, bins)
    eb = [np.mean(e[idx == k]) * 1000 if np.any(idx == k) else np.nan for k in range(1, 21)]
    rb = [np.mean((r.v - ocv(r.soc, r.T))[(idx == k) & rest]) * 1000 if np.any((idx == k) & rest) else np.nan
          for k in range(1, 21)]
    ax[2, col].plot(bins[:-1] + 0.025, eb, "o-", label="mean(sim - meas)")
    ax[2, col].plot(bins[:-1] + 0.025, rb, "s-", label="mean(V - OCV) where |I|<50mA")
    ax[2, col].axhline(0, color="k", lw=0.5)
    ax[2, col].legend(fontsize=7)
    ax[2, col].set(xlabel="SoC", ylabel="mV")
    print(f"{T}C {c}: RMSE {np.sqrt(np.mean(e**2))*1000:.1f} mV, mean {e.mean()*1000:.1f} mV, "
          f"T {r.T.min():.1f}..{r.T.max():.1f}")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "debug_ecm.png", dpi=110)
