"""Generate every report figure into results/figures/ from the saved results.

Colour is assigned by method and never changes between figures (palette slots 1-3 of the reference
palette are validated for all pairs; the ensemble/violet is always also distinguished by marker or label;
Coulomb counting is the neutral baseline grey).

    python scripts/make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from socest.api import P0_S, load_ekf_config
from socest.data import DT, RESULTS, TEMP_DIRS, TRAIN_TEMPS, load_cycle
from socest.ecm import simulate
from socest.ekf import coulomb_count, ekf
from socest.metrics import TARGET_RMSE
from socest.ml import estimate, load_model
from socest.ocv import OCV, c20_curves, hppc_rest_points
from socest.sensors import SCENARIOS, corrupt

FIG = RESULTS / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = dict(ekf="#2a78d6", lstm="#eb6834", gru="#1baf7a", ens="#6250d6", cc="#8a8984", truth=INK)
TEMPS = list(TEMP_DIRS)
TLAB = [f"{T}°C" + ("" if T in TRAIN_TEMPS else "\n(unseen)") for T in TEMPS]
SEL = json.loads((RESULTS / "ml_selected.json").read_text())
"""Selected ML configuration (chosen on validation by scripts/run_all.py)."""
ML_MAIN = SEL["tags"][0]
ML_LAB = f"LSTM-{SEL['config'][1]} ({SEL['config'][2]} s window)"
EKF_N = json.loads((RESULTS / "ekf_selected.json").read_text())["n_rc"]
"""ECM order selected on validation."""
EKF_LAB = f"EKF {EKF_N}-RC"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "lines.linewidth": 1.4, "savefig.dpi": 150,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})


def load(name):
    """Read a JSON file from results/, or None if it does not exist."""
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def rmse_by_T(res, method_prefix, scenario="realistic", key="rmse", agg=np.mean):
    """Aggregate a metric per temperature for one method and scenario (list in TEMPS order)."""
    return [agg([r[key] for r in res if r["scenario"] == scenario and r["temp"] == T
                 and r["method"].startswith(method_prefix)]) for T in TEMPS]


def target_line(ax):
    """Draw the 3 % acceptance threshold on an axis."""
    ax.axhline(TARGET_RMSE, color=INK2, lw=0.8, ls="--")
    ax.text(ax.get_xlim()[1], TARGET_RMSE, " 3 % target", va="center", ha="left", color=INK2, fontsize=8)


def fig_ocv(ocv):
    """OCV at 25 degC with the C/20 curves and HPPC rests; temperature dependence (ocv.png)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    (sd, vd), (sc, vc) = c20_curves()
    ax[0].plot(sd * 100, vd, color=C["cc"], lw=1, label="C/20 discharge")
    ax[0].plot(sc * 100, vc, color=C["cc"], lw=1, ls=":", label="C/20 charge (not used)")
    ax[0].plot(ocv.s * 100, ocv(ocv.s, 25), color=C["ekf"], label="OCV model, 25°C")
    p = hppc_rest_points(25)
    ax[0].plot(p[:, 0] * 100, p[:, 1], "o", ms=4, mfc="white", mec=INK, label="HPPC relaxed rests")
    ax[0].set(xlabel="SoC [%]", ylabel="Voltage [V]", title="OCV at 25°C", xlim=(-3, 101))
    ax[0].legend(fontsize=8)
    for T, mk in zip(TEMPS, "os^Dv"):
        p = hppc_rest_points(T)
        l, = ax[1].plot(ocv.s * 100, (ocv(ocv.s, T) - ocv(ocv.s, 25)) * 1000, lw=1.2,
                        color=INK if T in TRAIN_TEMPS else C["lstm"], ls="-" if T in TRAIN_TEMPS else "--")
        ax[1].plot(p[:, 0] * 100, (p[:, 1] - ocv(p[:, 0], 25)) * 1000, mk, ms=4, mfc="white", mec=l.get_color())
        ax[1].text(1, (ocv(0.01, T) - ocv(0.01, 25)) * 1000 + 2.5, f"{T}°C", fontsize=7.5,
                   va="bottom", color=l.get_color())   # curves are well separated at low SoC
    ax[1].set(xlabel="SoC [%]", ylabel="OCV(T) − OCV(25°C) [mV]", xlim=(0, 102), ylim=(-95, 15),
              title="Temperature dependence\n(black: training T, orange dashed: unseen T; markers: rests)")
    fig.savefig(FIG / "ocv.png")
    plt.close(fig)


def fig_voltage_fit(ocv):
    """Measured vs selected-ECM voltage on test cycles and voltage RMSE by temperature (voltage_fit.png)."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    th = np.load(RESULTS / f"ecm_theta_{EKF_N}rc.npy")
    for a, (T, c) in zip(ax[:2], [(25, "US06"), (-20, "UDDS")]):
        r = load_cycle(T, c)
        vs = simulate(th, r.soc, r.T, r.i, DT, ocv.table, ocv.s0, ocv.ds)
        sl = slice(int(1500 / DT), int(1800 / DT))
        a.plot(r.t[sl] - r.t[sl][0], r.v[sl], color=INK, lw=1, label="measured")
        a.plot(r.t[sl] - r.t[sl][0], vs[sl], color=C["ekf"], lw=1, label=f"{EKF_N}-RC model")
        a.set(xlabel="time [s]", ylabel="V", title=f"{T}°C {c} (test cycle), 300 s window")
        a.legend(fontsize=8)
    for n_rc, mk in [(1, "s"), (2, "o"), (3, "^")]:
        d = load(f"ecm_fit_{n_rc}rc.json")
        if d:
            vals = [np.mean([v for k, v in d["rmse_mV"].items() if k.startswith(f"{T}C_")]) for T in TEMPS]
            ax[2].plot(range(5), vals, mk + "-", color=[C["cc"], C["ekf"], INK][n_rc - 1], ms=5, label=f"{n_rc}-RC")
    ax[2].set_xticks(range(5), TLAB, fontsize=7)
    ax[2].set(ylabel="voltage RMSE [mV]", title="Model fit, all 9 cycles", ylim=(0, None))
    ax[2].legend(fontsize=8)
    fig.savefig(FIG / "voltage_fit.png")
    plt.close(fig)


def fig_trajectories(ocv):
    """SoC trajectories and errors with 3-sigma bands for three test cases (trajectories.png)."""
    theta, q, alpha = load_ekf_config(EKF_N)
    nets = [load_model(ML_MAIN)]
    a_ml = load(f"ml_{ML_MAIN}_eval.json")["alpha"]
    cases = [(25, "UDDS", "realistic"), (-10, "HWFET", "realistic"), (-20, "US06", "realistic")]
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 2.5 * len(cases)), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for (T, c, scn), (a0, a1) in zip(cases, axes):
        run = load_cycle(T, c)
        v, i, Tm, soc0 = corrupt(run, SCENARIOS[scn], 0)
        cc = coulomb_count(i, DT, soc0)
        se, sg, _ = ekf(theta, v, i, Tm, DT, ocv.table, ocv.s0, ocv.ds, soc0, P0_S, *q)
        sm, sgm = estimate(nets, v, i, Tm)
        h = run.t / 3600
        a0.plot(h, run.soc * 100, color=INK, lw=1.8, label="truth")
        a0.plot(h, cc * 100, color=C["cc"], lw=1, label="Coulomb counting")
        a0.plot(h, se * 100, color=C["ekf"], lw=1, label=EKF_LAB)
        a0.plot(h, sm * 100, color=C["lstm"], lw=1, label=ML_LAB)
        a0.set(ylabel="SoC [%]", title=f"{T}°C {c}{'' if T in TRAIN_TEMPS else ' (unseen T)'} — {scn} sensors")
        a1.fill_between(h, -300 * alpha * sg, 300 * alpha * sg, color=C["ekf"], alpha=0.15, lw=0, label="EKF ±3σ")
        a1.fill_between(h, -300 * a_ml * sgm, 300 * a_ml * sgm, color=C["lstm"], alpha=0.12, lw=0, label="LSTM ±3σ")
        a1.plot(h, (se - run.soc) * 100, color=C["ekf"], lw=0.9, label="EKF error")
        a1.plot(h, (sm - run.soc) * 100, color=C["lstm"], lw=0.9, label="LSTM error")
        a1.axhline(0, color=INK2, lw=0.6)
        a1.set(ylabel="error [% SoC]", ylim=(-12, 12))
        for a in (a0, a1):
            a.legend(fontsize=7, ncol=2, loc="lower left" if a is a0 else "upper right")
    axes[-1, 0].set_xlabel("time [h]")
    axes[-1, 1].set_xlabel("time [h]")
    fig.tight_layout()
    fig.savefig(FIG / "trajectories.png")
    plt.close(fig)


def fig_accuracy(ekf_res, ml):
    """Test RMSE and worst settled error by temperature (accuracy_by_temperature.png)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(5)
    seeds = [load(f"ml_{t}_eval.json") for t in SEL["seeds"]]
    seeds = [s["results"] for s in seeds if s]
    series = [("Coulomb counting", rmse_by_T(ekf_res, "coulomb"), C["cc"], None),
              (EKF_LAB, rmse_by_T(ekf_res, "ekf"), C["ekf"], None),
              (f"{ML_LAB}, mean of 3 seeds", np.mean([rmse_by_T(s, "lstm") for s in seeds], 0), C["lstm"],
               np.ptp([rmse_by_T(s, "lstm") for s in seeds], 0) / 2)]
    w = 0.26
    for k, (lab, vals, col, err) in enumerate(series):
        ax[0].bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=lab,
                  yerr=err, ecolor=INK2, capsize=2, error_kw=dict(lw=0.8))
    ax[0].set_xticks(x, TLAB, fontsize=8)
    ax[0].set(ylabel="RMSE [% SoC]", title="Accuracy on test cycles, realistic sensors\n(error bars: seed range)")
    target_line(ax[0])
    ax[0].legend(fontsize=8, loc="upper left")
    for k, (lab, pre, col, res) in enumerate([(EKF_LAB, "ekf", C["ekf"], ekf_res),
                                              (f"{ML_LAB}, seed 0", "lstm", C["lstm"], ml)]):
        vals = rmse_by_T(res, pre, key="maxabs_settled", agg=np.max)
        ax[1].bar(x + (k - 0.5) * 0.36, vals, 0.33, color=col, label=lab)
    ax[1].set_xticks(x, TLAB, fontsize=8)
    ax[1].set(ylabel="worst |error| after 10 min [% SoC]", title="Worst-case error (max over the 3 test cycles)")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / "accuracy_by_temperature.png")
    plt.close(fig)


def fig_robustness(ekf_res, ml):
    """RMSE heatmap by sensor scenario and method (robustness.png)."""
    methods = [("Coulomb counting", ekf_res, "coulomb"), (EKF_LAB, ekf_res, "ekf"), (ML_LAB, ml, "lstm")]
    scen = list(SCENARIOS)
    M = np.array([[np.mean([r["rmse"] for r in res if r["scenario"] == s and r["method"].startswith(p)])
                   for s in scen] for _, res, p in methods])
    fig, ax = plt.subplots(figsize=(8, 2.4))
    im = ax.imshow(np.minimum(M, 10), cmap="Blues", vmin=0, vmax=10, aspect="auto")
    for (a, b), val in np.ndenumerate(M):
        ax.text(b, a, f"{val:.1f}", ha="center", va="center", color="white" if val > 6 else INK, fontsize=9)
    ax.set_xticks(range(len(scen)), scen, fontsize=8)
    ax.set_yticks(range(3), [m[0] for m in methods])
    ax.grid(False)
    ax.set_title("Test RMSE [% SoC] by sensor scenario (mean over 5 temperatures × 3 cycles)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="RMSE (capped at 10)")
    fig.savefig(FIG / "robustness.png")
    plt.close(fig)


def fig_sensitivity(sens):
    """Covariance surface, sampling rate, parameter tornado, LSTM settings (sensitivity.png)."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    g = sens["covariance_surface"]
    qv = g["chosen"][1]
    qs = sorted({t["q"][0] for t in g["grid"]})
    rs = sorted({t["q"][2] for t in g["grid"]})
    Z = np.full((len(qs), len(rs)), np.nan)
    Cv = np.full_like(Z, np.nan)
    for t in g["grid"]:
        if t["q"][1] == qv:
            Z[qs.index(t["q"][0]), rs.index(t["q"][2])] = t["rmse"]
            Cv[qs.index(t["q"][0]), rs.index(t["q"][2])] = t["coverage"]
    im = ax[0, 0].imshow(Z, cmap="Blues_r", aspect="auto", origin="lower")
    for (a, b), val in np.ndenumerate(Z):
        ax[0, 0].text(b, a, f"{val:.1f}\n{Cv[a, b]*100:.0f}%", ha="center", va="center", fontsize=6.5,
                      color="white" if val < np.nanmedian(Z) else INK)
    a, b = qs.index(g["chosen"][0]), rs.index(g["chosen"][2])
    ax[0, 0].add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=C["lstm"], lw=2))
    ax[0, 0].set_xticks(range(len(rs)), [f"{np.sqrt(r)*1000:.0f}" for r in rs])
    ax[0, 0].set_yticks(range(len(qs)), [f"{q:.0e}" for q in qs])
    ax[0, 0].set(xlabel="√r: assumed voltage error [mV]", ylabel="q_s per 0.1 s step",
                 title=f"EKF tuning surface on validation (q_v = {qv:g})\nRMSE % / 3σ coverage; box = chosen")
    ax[0, 0].grid(False)
    sr = {float(k): v for k, v in sens["sampling_rate_hz"].items()}   # JSON keys are strings
    hz = sorted(sr)
    ax[0, 1].plot(hz, [sr[h] for h in hz], "o-", color=C["ekf"], ms=5)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xticks(hz, [f"{h:g}" for h in hz])
    ax[0, 1].set(xlabel="EKF sample rate [Hz] (same parameters, exact ZOH)", ylabel="test RMSE [% SoC]",
                 title="Sampling-rate sensitivity", ylim=(0, None))
    p = sens["perturbation"]["delta_rmse"]
    items = sorted(p.items(), key=lambda kv: abs(kv[1]))
    ax[1, 0].barh(range(len(items)), [v for _, v in items],
                  color=[C["lstm"] if v > 0 else C["ekf"] for _, v in items], height=0.7)
    ax[1, 0].set_yticks(range(len(items)), [k for k, _ in items], fontsize=8)
    ax[1, 0].axvline(0, color=INK2, lw=0.8)
    ax[1, 0].set(xlabel="change in test RMSE [% points]",
                 title=f"One-at-a-time perturbation (base {sens['perturbation']['base_rmse']:.2f} %)")
    ml = sens["ml"]
    groups = [("window", [("lstm32_c100_s0", "100 s"), ("lstm32_c250_s0", "250 s"), ("lstm32_c500_s0", "500 s"),
                          ("lstm32_c1000_s0", "1000 s")]),
              ("hidden", [("lstm16_c500_s0", "16"), ("lstm32_c500_s0", "32"), ("lstm64_c500_s0", "64")])]
    xs, labels = [], []
    pos = 0
    for gname, items in groups:
        for tag, lab in items:
            if tag in ml:
                ax[1, 1].bar(pos, ml[tag]["rmse"], 0.7, color=C["lstm"] if gname == "window" else C["gru"])
                labels.append(f"{lab}")
                xs.append(pos)
            pos += 1
        pos += 0.6
    ax[1, 1].set_xticks(xs, labels, fontsize=8)
    ax[1, 1].set(ylabel="test RMSE [% SoC]", title="LSTM: BPTT window (orange) and hidden size (aqua)")
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png")
    plt.close(fig)


def fig_pareto(perf, sens):
    """Accuracy vs compute with the Pareto front (pareto.png)."""
    models = perf["models"]
    rc = sens["n_rc"]
    pts = [(f"EKF {n}-RC", models[f"ekf_{n}rc"]["flops_per_s"], rc[str(n)]["soc_rmse"], C["ekf"], "o")
           for n in (1, 2, 3) if f"ekf_{n}rc" in models and str(n) in rc]
    # the sampling-rate study showed the selected EKF loses nothing at 2 Hz: include it as the optimised point
    sr = {float(k): v for k, v in sens["sampling_rate_hz"].items()}
    if 2.0 in sr:
        pts.append((f"EKF {EKF_N}-RC at 2 Hz", models[f"ekf_{EKF_N}rc"]["flops_per_s"] / 5, sr[2.0], C["ekf"], "o"))
    for tag, key, lab, col, mk in [("lstm16_c500_s0", "lstm16", "LSTM-16", C["lstm"], "s"),
                                   ("lstm32_c500_s0", "lstm32", "LSTM-32 (500 s)", C["lstm"], "s"),
                                   ("lstm32_c100_s0", "lstm32", "LSTM-32 (100 s, selected)", C["lstm"], "s"),
                                   ("lstm64_c500_s0", "lstm64", "LSTM-64", C["lstm"], "s"),
                                   ("gru32_c500_s0", "gru32", "GRU-32", C["gru"], "D")]:
        if tag in sens["ml"]:
            pts.append((lab, models[key]["flops_per_s"], sens["ml"][tag]["rmse"], col, mk))
    ens = SEL.get("ensemble")
    if ens in sens["ml"]:
        pts.append(("3-seed ensemble (100 s)", 3 * models[f"lstm{SEL['config'][1]}"]["flops_per_s"],
                    sens["ml"][ens]["rmse"], C["ens"], "*"))
    # hand-placed label offsets [points] where labels would otherwise collide
    nudge = {"LSTM-32 (500 s)": (6, 4), "LSTM-32 (100 s, selected)": (6, -12), f"EKF {EKF_N}-RC": (6, 6),
             "LSTM-64": (6, 4), "3-seed ensemble (100 s)": (-20, -14)}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for lab, x, y, col, mk in pts:
        ax.plot(x, y, mk, color=col, ms=14 if mk == "*" else 7, mec="white", mew=1.2)
        ax.annotate(lab, (x, y), xytext=nudge.get(lab, (6, 4)), textcoords="offset points", fontsize=8, color=INK)
    # Pareto front: no other point is both cheaper and more accurate
    front = sorted([p for p in pts if not any(q[1] <= p[1] and q[2] < p[2] and q != p for q in pts)],
                   key=lambda p: p[1])
    ax.plot([p[1] for p in front], [p[2] for p in front], color=INK2, lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set(xlabel="compute [FLOP per second of driving] (log)", ylabel="test RMSE [% SoC], realistic sensors",
           title="Accuracy vs compute (dashed: Pareto front)\nblue circles = physics-based, interpretable states;"
                 " others = black-box", ylim=(0, None))
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)


def fig_before_after():
    """Before/after comparisons of the main design changes (before_after.png)."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(5)
    d10, d2 = load("ekf_2rc_d10_eval.json"), load("ekf_2rc_eval.json")
    for k, (lab, d, col) in enumerate([("residual at 1 Hz (before)", d10, C["cc"]), ("at 5 Hz (after)", d2, C["ekf"])]):
        ax[0].bar(x + (k - 0.5) * 0.36, rmse_by_T(d["results"], "ekf"), 0.33, color=col, label=lab)
    ax[0].set_xticks(x, TLAB, fontsize=7)
    ax[0].set(ylabel="EKF test RMSE [% SoC]", title="2-RC ECM fit: residual sampling", ylim=(0, 4.6))
    target_line(ax[0])
    ax[0].legend(fontsize=7, loc="upper left")
    g = d2["tuning"]
    best_rmse = min(g, key=lambda t: t["rmse"])
    chosen = next(t for t in g if tuple(t["q"]) == tuple(d2["q"]))
    vals = [(best_rmse["rmse"], best_rmse["coverage"]), (chosen["rmse"], chosen["coverage"])]
    ax[1].bar([0, 1], [v[1] * 100 for v in vals], 0.5, color=[C["cc"], C["ekf"]])
    for k, (r_, c_) in enumerate(vals):
        ax[1].text(k, c_ * 100 + 2, f"RMSE {r_:.2f} %", ha="center", fontsize=8, color=INK)
    ax[1].set_xticks([0, 1], ["RMSE-only tuning", "coverage-constrained"])
    ax[1].set(ylabel="3σ coverage on validation [%]", ylim=(0, 110), title="EKF tuning objective")
    v0, v1, v2 = (load("ml_lstm32_c500_v0_eval.json"), load("ml_lstm32_c500_s0_eval.json"),
                  load(f"ml_{ML_MAIN}_eval.json"))
    if v0 and v1 and v2:
        w = 0.27
        for k, (lab, d, col) in enumerate([("first LSTM", v0, C["cc"]), ("+ β-NLL, lr 1e-3", v1, C["gru"]),
                                           (f"+ {SEL['config'][2]} s BPTT window (selected)", v2, C["lstm"])]):
            ax[2].bar(x + (k - 1) * w, rmse_by_T(d["results"], "lstm"), w - 0.03, color=col, label=lab)
        ax[2].set_xticks(x, TLAB, fontsize=7)
        ax[2].set(ylabel="LSTM test RMSE [% SoC]", title="LSTM training changes", ylim=(0, 4.6))
        target_line(ax[2])
        ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "before_after.png")
    plt.close(fig)


def main():
    """Generate every figure."""
    FIG.mkdir(exist_ok=True)
    ocv = OCV()
    ekf_res = load(f"ekf_{EKF_N}rc_eval.json")["results"]
    ml = load(f"ml_{ML_MAIN}_eval.json")["results"]
    sens, perf = load("sensitivity.json"), load("perf.json")
    fig_ocv(ocv)
    fig_voltage_fit(ocv)
    fig_trajectories(ocv)
    fig_accuracy(ekf_res, ml)
    fig_robustness(ekf_res, ml)
    fig_before_after()
    if sens:
        fig_sensitivity(sens)
    if sens and perf:
        fig_pareto(perf, sens)
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("ti", RESULTS.parent / "tests" / "test_integrator.py")
    ti = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ti)
    ti.figure()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
