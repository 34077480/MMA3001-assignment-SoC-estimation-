"""Reproduce every result, figure and document from the raw data, in order.

    python scripts/run_all.py            full pipeline (~1.5-2 h on an 8-core laptop, CPU only)
    python scripts/run_all.py --skip-fit --skip-train   reuse fitted ECMs and trained networks (~10 min)

Steps: fit ECMs (1/2/3-RC) -> tune + evaluate EKFs -> train networks (3 LSTM seeds, GRU, window and
size variants) -> evaluate networks and the ensemble -> sensitivity -> performance -> figures -> tests
(+ HTML report) -> API documentation (pdoc).
"""
import argparse
import json
import subprocess
import sys

from socest.data import RESULTS, ROOT

PY = sys.executable
SEEDS = [0, 1, 2]
CONFIGS = [("lstm", 32, 500), ("gru", 32, 500), ("lstm", 32, 100), ("lstm", 32, 250), ("lstm", 32, 1000),
           ("lstm", 16, 500), ("lstm", 64, 500)]
"""Network configurations trained with seed 0; the one with the best validation RMSE is then trained
with SEEDS 1 and 2 as well (repeatability) and becomes the selected ML model."""


def select_ekf():
    """Pick the ECM order whose tuned EKF has the lowest validation RMSE; write results/ekf_selected.json."""
    val = {}
    for n in (1, 2, 3):
        d = json.loads((RESULTS / f"ekf_{n}rc_eval.json").read_text())
        val[n] = next(t for t in d["tuning"] if tuple(t["q"]) == tuple(d["q"]))["rmse"]
    best = min(val, key=val.get)
    (RESULTS / "ekf_selected.json").write_text(json.dumps(dict(n_rc=best, validation_rmse=val), indent=1))


def select_config():
    """Configuration (kind, hidden, chunk) with the lowest validation RMSE, from the seed-0 runs."""
    best = min(CONFIGS, key=lambda c: json.loads(
        (RESULTS / f"ml_{c[0]}{c[1]}_c{c[2]}_s0_train.json").read_text())["best_val"])
    return best


def sh(*args):
    """Run one pipeline step, stopping on failure."""
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def main():
    """Run the whole pipeline."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    if not a.skip_fit:
        sh(PY, "scripts/fit_ecm.py", 1, 2, 3)
    for n_rc in (1, 2, 3):
        sh(PY, "scripts/run_ekf.py", n_rc)
    select_ekf()
    train = lambda k, h, c, s: sh(PY, "scripts/train_ml.py", "--kind", k, "--hidden", h, "--chunk", c, "--seed", s)
    if not a.skip_train:
        for k, h, c in CONFIGS:
            train(k, h, c, 0)
    k, h, c = select_config()          # model selection on validation only
    if not a.skip_train:
        for s in SEEDS[1:]:
            train(k, h, c, s)
    tags = [f"{kk}{hh}_c{cc}_s0" for kk, hh, cc in CONFIGS] + [f"{k}{h}_c{c}_s{s}" for s in SEEDS[1:]]
    for t in tags:
        sh(PY, "scripts/run_ml.py", t)
    sel = [f"{k}{h}_c{c}_s{s}" for s in SEEDS]
    sh(PY, "scripts/run_ml.py", *sel, "--name", f"ens3_{k}{h}_c{c}")
    # the API uses the selected configuration's seed-0 network (a single net: 1/3 of the ensemble's cost)
    alpha = json.loads((RESULTS / f"ml_{sel[0]}_eval.json").read_text())["alpha"]
    (RESULTS / "ml_selected.json").write_text(json.dumps(dict(tags=[sel[0]], alpha=alpha, config=[k, h, c],
                                                              seeds=sel, ensemble=f"ens3_{k}{h}_c{c}"), indent=1))
    sh(PY, "scripts/sensitivity.py")
    sh(PY, "scripts/perf.py")
    sh(PY, "scripts/make_figures.py")
    sh(PY, "-m", "pytest", "--html=docs/test_report.html", "--self-contained-html")
    sh(PY, "-m", "pdoc", "socest", "-o", "docs/api", "--docformat", "numpy")


if __name__ == "__main__":
    main()
