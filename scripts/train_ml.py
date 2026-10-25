"""Train one recurrent SoC network and save it as ``results/ml_<kind><hidden>_c<chunk>_s<seed>.pt``.

    python scripts/train_ml.py --kind lstm --hidden 32 --chunk 500 --seed 0
"""
import argparse
import json

import torch

from socest.data import RESULTS
from socest.ml import train


def main():
    """Parse arguments, train one network and save it."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=500, help="truncated-BPTT window [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, info = train(a.kind, a.hidden, a.chunk, epochs=a.epochs, seed=a.seed, verbose=False)
    tag = f"{a.kind}{a.hidden}_c{a.chunk}_s{a.seed}"
    torch.save(model.state_dict(), RESULTS / f"ml_{tag}.pt")
    (RESULTS / f"ml_{tag}_train.json").write_text(json.dumps(info, indent=1))
    print(f"saved {tag}: best val RMSE {info['best_val']:.2f}%, {info['train_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
