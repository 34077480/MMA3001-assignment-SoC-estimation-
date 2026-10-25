"""Recurrent (LSTM/GRU) SoC estimator with a Gaussian output head (mean + log-variance).

Inputs (V, I, T) at 1 Hz (10 Hz block-averaged), normalised. The network runs statefully over a whole
drive cycle, so at deployment it costs one recurrent step per second of driving.

Training details that matter for a fair comparison with the EKF:
* sequences start at a random point in the cycle, so the network must infer SoC from the measurements
  instead of memorising "cycles start full";
* the same sensor-corruption model used to tune the EKF is applied as augmentation;
* truncated BPTT over whole cycles (hidden state carried between chunks), matching stateful inference;
* early stopping on the LA92 validation cycle; nothing from the test cycles or 10C/-10C is used.

    python src/ml.py [lstm|gru] [hidden] [chunk]
"""
import json
import sys
import time

import numpy as np
import torch
from torch import nn

from data import ROOT, TRAIN_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle

OUT = ROOT / "results"
FS_DECIM = 10            # 10 Hz -> 1 Hz
NORM = dict(v=(3.6, 0.4), i=(0.0, 10.0), T=(5.0, 20.0))


def to_1hz(x, k=FS_DECIM):
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(1)


def features(v, i, T):
    return np.stack([(v - NORM["v"][0]) / NORM["v"][1], (i - NORM["i"][0]) / NORM["i"][1],
                     (T - NORM["T"][0]) / NORM["T"][1]], -1).astype(np.float32)


class SocNet(nn.Module):
    def __init__(self, kind="lstm", hidden=32, layers=1):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn(3, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        out = self.head(y)
        mu = out[..., 0] + 0.5
        logvar = out[..., 1].clamp(-14, 2)
        return mu, logvar, h


def augment(r, rng, p_clean=0.25):
    """Random sensor corruption (same family as sensors.SCENARIOS), at 10 Hz before downsampling."""
    n = len(r.t)
    if rng.random() < p_clean:
        return r.v, r.i, r.T
    v = r.v + rng.uniform(0, 0.01) * rng.standard_normal(n)
    i = (1 + rng.uniform(-0.01, 0.01)) * r.i + rng.uniform(-0.05, 0.05) + rng.uniform(0, 0.1) * rng.standard_normal(n)
    T = r.T + rng.uniform(0, 0.5) * rng.standard_normal(n)
    return v, i, T


def make_batch(runs, rng, max_start_frac=0.6):
    xs, ys = [], []
    for r in runs:
        v, i, T = augment(r, rng)
        x = features(to_1hz(v), to_1hz(i), to_1hz(T))
        y = to_1hz(r.soc).astype(np.float32)
        s = int(rng.uniform(0, max_start_frac) * len(y))
        xs.append(x[s:])
        ys.append(y[s:])
    L = max(len(y) for y in ys)
    X = np.zeros((len(xs), L, 3), np.float32)
    Y = np.zeros((len(xs), L), np.float32)
    M = np.zeros((len(xs), L), np.float32)
    for k, (x, y) in enumerate(zip(xs, ys)):
        X[k, :len(y)], Y[k, :len(y)], M[k, :len(y)] = x, y, 1
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


def gaussian_nll(mu, logvar, y, m, warm, beta=0.5):
    """beta-NLL (Seitzer et al., 2022): NLL weighted by var^beta (no gradient through the weight).
    Plain NLL (beta=0) lets the network 'explain away' hard regions with a large variance and stop fitting
    the mean there; beta=0.5 keeps the mean accurate while the variance head still learns."""
    # skip the first `warm` steps of every sequence: the network needs a little data before it can know SoC
    m = m.clone()
    m[:, :warm] = 0
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp()) * logvar.detach().exp() ** beta
    return (nll * m).sum() / m.sum(), (((y - mu) ** 2) * m).sum() / m.sum()


@torch.no_grad()
def predict(model, v, i, T):
    """Run statefully over a full 10 Hz record; returns 1 Hz (mu, sigma)."""
    x = torch.from_numpy(features(to_1hz(v), to_1hz(i), to_1hz(T)))[None]
    mu, logvar, _ = model(x)
    return mu[0].numpy(), (0.5 * logvar[0]).exp().numpy()


def train(kind="lstm", hidden=32, chunk=500, epochs=600, warm=60, seed=0, lr=1e-3, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_runs = [load_cycle(T, c) for T in TRAIN_TEMPS for c in TRAIN_CYCLES]
    val_runs = [load_cycle(T, c) for T in TRAIN_TEMPS for c in VAL_CYCLES]
    model = SocNet(kind, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        X, Y, M = make_batch(train_runs, rng)
        h = None
        tot = 0.0
        # first epochs: MSE only (stable start), then Gaussian NLL for the uncertainty head
        use_nll = ep >= epochs // 5
        for a in range(0, X.shape[1], chunk):
            mu, logvar, h = model(X[:, a:a + chunk], h)
            h = tuple(z.detach() for z in h) if isinstance(h, tuple) else h.detach()
            nll, mse = gaussian_nll(mu, logvar, Y[:, a:a + chunk], M[:, a:a + chunk], warm if a == 0 else 0)
            loss = nll if use_nll else mse
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(mse.detach())
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            model.eval()
            e = []
            for r in val_runs:
                mu, _ = predict(model, r.v, r.i, r.T)
                e.append(np.sqrt(np.mean((mu - to_1hz(r.soc)) ** 2)) * 100)
            val = float(np.mean(e))
            hist.append(dict(epoch=ep + 1, val_rmse=val))
            if val < best and use_nll:
                best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep {ep+1:4d}  train mse-rmse {np.sqrt(tot/max(1, X.shape[1]//chunk))*100:5.2f}%  "
                      f"val RMSE {val:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, dict(kind=kind, hidden=hidden, chunk=chunk, epochs=epochs, best_val=best, history=hist,
                       train_time_s=time.time() - t0)


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "lstm"
    hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    torch.set_num_threads(4)
    model, info = train(kind, hidden, chunk, seed=seed)
    tag = f"{kind}{hidden}_c{chunk}_s{seed}"
    torch.save(model.state_dict(), OUT / f"ml_{tag}.pt")
    (OUT / f"ml_{tag}_train.json").write_text(json.dumps(info, indent=1))
    print(f"saved {tag}: best val RMSE {info['best_val']:.2f}%, {info['train_time_s']:.0f}s")
