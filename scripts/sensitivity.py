"""Sensitivity analyses. All SoC numbers are test-set RMSE [% SoC] under the 'realistic' sensor scenario,
mean over the 15 test runs (3 EPA cycles x 5 temperatures), unless stated otherwise.

1. EKF noise covariances: the validation tuning surface (q_s, r) and its coverage.
2. Number of RC pairs (1, 2, 3): voltage RMSE and EKF SoC RMSE.
3. Sampling rate: the EKF run at 10, 5, 2, 1 and 0.5 Hz with the *same* parameters. The ZOH
   discretisation is exact, so no refit is needed; process noise is scaled to stay constant per second.
4. One-at-a-time parameter perturbation of the selected EKF (which inputs matter most): each resistance and
   time constant x0.8 / x1.2, the OCV curve +-10 mV, and the capacity +-5 % (implemented as an equivalent
   current-gain error).
5. Recurrent network: truncated-BPTT window (100-1000 s), hidden size (16-64) and seed repeatability.

    python scripts/sensitivity.py        writes results/sensitivity.json
"""
import json

import numpy as np

from socest.api import P0_S, load_ekf_config, selected_n_rc
from socest.ecm import n_rc_of
from socest.data import DT, RESULTS, TEMP_DIRS, TEST_CYCLES, Run, load_cycle
from socest.ekf import ekf
from socest.metrics import soc_metrics
from socest.ocv import OCV
from socest.sensors import SCENARIOS, corrupt

SC = SCENARIOS["realistic"]


def test_runs():
    """Load the 15 held-out test runs."""
    return [load_cycle(T, c) for T in TEMP_DIRS for c in TEST_CYCLES]


def ekf_rmse(theta, q, runs, ocv, table=None, i_gain=0.0, decim=1):
    """Mean test RMSE of the EKF; optional OCV table override, extra current gain, and decimation."""
    tab = ocv.table if table is None else table
    out = []
    for r in runs:
        v, i, T, soc0 = corrupt(r, SC, 0)
        i = i * (1 + i_gain)
        rr = r
        if decim > 1:   # block-average: preserves charge exactly for the Coulomb-counting part
            m = len(v) // decim * decim
            blk = lambda x: x[:m].reshape(-1, decim).mean(1)
            v, i, T = blk(v), blk(i), blk(T)
            idx = np.arange(decim - 1, m, decim)          # estimate available at the end of each block
            rr = Run(r.name, r.temp, r.t[idx], r.v[idx], r.i[idx], r.T[idx], r.ah[idx])
        dt = DT * decim
        soc, _, _ = ekf(theta, v, i, T, dt, tab, ocv.s0, ocv.ds, soc0, P0_S, q[0] * decim, q[1] * decim, q[2])
        out.append(soc_metrics(soc, rr)["rmse"])
    return float(np.mean(out)), [float(x) for x in out]


def main():
    """Run all sensitivity studies."""
    ocv = OCV()
    runs = test_runs()
    res = {}

    # 1. covariance surface from the validation tuning
    cfg = json.loads((RESULTS / f"ekf_{selected_n_rc()}rc_eval.json").read_text())
    res["covariance_surface"] = dict(chosen=cfg["q"], grid=cfg["tuning"])

    # 2. number of RC pairs
    rc = {}
    for n_rc in (1, 2, 3):
        f = RESULTS / f"ekf_{n_rc}rc_eval.json"
        g = RESULTS / f"ecm_fit_{n_rc}rc.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        rr = [r for r in d["results"] if r["scenario"] == "realistic" and r["method"].startswith("ekf")]
        rc[n_rc] = dict(soc_rmse=float(np.mean([r["rmse"] for r in rr])),
                        soc_rmse_by_T={T: float(np.mean([r["rmse"] for r in rr if r["temp"] == T])) for T in TEMP_DIRS},
                        coverage=float(np.mean([r["coverage_3sig"] for r in rr])))
        if g.exists():
            rc[n_rc]["voltage_rmse_mV"] = float(np.mean(list(json.loads(g.read_text())["rmse_mV"].values())))
    res["n_rc"] = rc
    print("n_rc:", {k: round(v["soc_rmse"], 2) for k, v in rc.items()})

    theta, q, _ = load_ekf_config()          # validation-selected ECM order

    # 3. sampling rate
    base, _ = ekf_rmse(theta, q, runs, ocv)
    rates = {}
    for decim in (1, 2, 5, 10, 20):
        rates[10 / decim] = ekf_rmse(theta, q, runs, ocv, decim=decim)[0]
    res["sampling_rate_hz"] = rates
    print("sampling rate:", {k: round(v, 2) for k, v in rates.items()})

    # 4. one-at-a-time perturbations
    names = ["R0"] + [f"{p}{m+1}" for m in range(n_rc_of(theta)) for p in ("R", "tau")]
    pert = {}
    for row, nm in enumerate(names):
        for f in (0.8, 1.2):
            th = theta.copy()
            th[row, :-2] += np.log(f)     # scales the parameter by f at every SoC and temperature
            pert[f"{nm} x{f}"] = ekf_rmse(th, q, runs, ocv)[0] - base
    for dv in (-0.01, 0.01):
        tab = ocv.table.copy()
        tab[0] += dv
        pert[f"OCV {dv*1000:+.0f} mV"] = ekf_rmse(theta, q, runs, ocv, table=tab)[0] - base
    for dq in (-0.05, 0.05):
        # assuming capacity Q(1+dq) is equivalent to scaling the measured current by 1/(1+dq)
        pert[f"capacity {dq*100:+.0f}%"] = ekf_rmse(theta, q, runs, ocv, i_gain=1 / (1 + dq) - 1)[0] - base
    res["perturbation"] = dict(base_rmse=base, delta_rmse=pert)
    print(f"perturbation (base {base:.2f}%):")
    for k, v in sorted(pert.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:16s} {v:+.2f}")

    # 5. recurrent network settings (from the evaluated models)
    ml = {}
    for f in sorted(RESULTS.glob("ml_*_eval.json")):
        d = json.loads(f.read_text())
        rr = [r for r in d["results"] if r["scenario"] == "realistic"]
        name = f.name[3:-10]
        ml[name] = dict(rmse=float(np.mean([r["rmse"] for r in rr])),
                        maxabs_settled=float(np.max([r["maxabs_settled"] for r in rr])),
                        rmse_by_T={T: float(np.mean([r["rmse"] for r in rr if r["temp"] == T])) for T in TEMP_DIRS})
        tj = RESULTS / f"ml_{name}_train.json"
        if tj.exists():
            ml[name]["best_val"] = json.loads(tj.read_text())["best_val"]
    res["ml"] = ml
    print("ml:", {k: round(v["rmse"], 2) for k, v in ml.items()})
    (RESULTS / "sensitivity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
