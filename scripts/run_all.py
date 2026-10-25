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
MAIN_SEEDS = [0, 1, 2]
VARIANTS = [("gru", 32, 500, 0), ("lstm", 32, 100, 0), ("lstm", 32, 250, 0), ("lstm", 32, 1000, 0),
            ("lstm", 16, 500, 0), ("lstm", 64, 500, 0)]


def sh(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-fit", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()
    if not a.skip_fit:
        sh(PY, "scripts/fit_ecm.py", 1, 2, 3)
    for n_rc in (1, 2, 3):
        sh(PY, "scripts/run_ekf.py", n_rc)
    models = [("lstm", 32, 500, s) for s in MAIN_SEEDS] + VARIANTS
    if not a.skip_train:
        for kind, h, c, s in models:
            sh(PY, "scripts/train_ml.py", "--kind", kind, "--hidden", h, "--chunk", c, "--seed", s)
    for kind, h, c, s in models:
        sh(PY, "scripts/run_ml.py", f"{kind}{h}_c{c}_s{s}")
    ens = [f"lstm32_c500_s{s}" for s in MAIN_SEEDS]
    sh(PY, "scripts/run_ml.py", *ens, "--name", "ens3_lstm32")
    # the selected ML model for the API: single LSTM-32 seed 0 (see report for why not the ensemble)
    alpha = json.loads((RESULTS / "ml_lstm32_c500_s0_eval.json").read_text())["alpha"]
    (RESULTS / "ml_selected.json").write_text(json.dumps(dict(tags=["lstm32_c500_s0"], alpha=alpha), indent=1))
    sh(PY, "scripts/sensitivity.py")
    sh(PY, "scripts/perf.py")
    sh(PY, "scripts/make_figures.py")
    sh(PY, "-m", "pytest", "--html=docs/test_report.html", "--self-contained-html")
    sh(PY, "-m", "pdoc", "socest", "-o", "docs/api", "--docformat", "numpy")


if __name__ == "__main__":
    main()
