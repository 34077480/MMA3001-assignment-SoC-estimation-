"""Recurrent-network (LSTM/GRU) SoC estimator with a Gaussian output head.

Design
------
* **Inputs:** voltage, current and cell temperature, block-averaged from 10 Hz to 1 Hz and normalised
  with fixed constants (``NORM``) so no statistics leak from validation or test data.
* **Outputs:** per time step, a mean SoC and a log-variance (heteroscedastic uncertainty).
* **Deployment:** the network runs statefully over a whole drive cycle, one recurrent step per second.
  It needs **no initial SoC**, unlike Coulomb counting and the EKF.

Training choices (for a fair comparison with the EKF)
-----------------------------------------------------
* Each training sequence starts at a random point (up to 60 % into the cycle), so the network must infer
  SoC from the measurements instead of memorising "cycles start full".
* Random sensor corruption from the same family as :mod:`socest.sensors` is applied as augmentation; the
  EKF was tuned against these scenarios too.
* Truncated back-propagation through time over whole cycles: the hidden state is carried between chunks
  (detached), which matches stateful inference.
* Loss: MSE for the first 20 % of epochs (stable start), then beta-NLL (Seitzer et al., 2022).
* Model selection (early stopping) on the LA92 validation cycle only.
"""
import time

import numpy as np
import torch
from torch import nn

from .data import RESULTS, TRAIN_CYCLES, TRAIN_TEMPS, VAL_CYCLES, load_cycle

FS_DECIM = 10
"""Decimation factor: 10 Hz input -> 1 Hz network rate."""
NORM = dict(v=(3.6, 0.4), i=(0.0, 10.0), T=(5.0, 20.0))
"""Fixed (offset, scale) per input channel, chosen from physical ranges, not from data statistics."""


def to_1hz(x, k=FS_DECIM):
    """Block-average a 10 Hz signal to 1 Hz (drops a trailing partial block)."""
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(1)


def to_10hz(x, n, k=FS_DECIM):
    """Map a 1 Hz estimate back onto an n-sample 10 Hz grid, causally.

    The value computed from block j becomes available at the last sample of block j and is held until
    the next one, so no estimate ever uses future measurements.
    """
    y = np.repeat(x, k)
    y = np.concatenate([np.full(k - 1, y[0]), y])[:n]
    return np.concatenate([y, np.full(n - len(y), y[-1])]) if len(y) < n else y


def features(v, i, T):
    """Normalised network input of shape (n, 3) from 1 Hz voltage [V], current [A] and temperature [degC]."""
    return np.stack([(v - NORM["v"][0]) / NORM["v"][1], (i - NORM["i"][0]) / NORM["i"][1],
                     (T - NORM["T"][0]) / NORM["T"][1]], -1).astype(np.float32)


class SocNet(nn.Module):
    """Single-layer LSTM or GRU followed by a linear head giving (mean SoC, log-variance).

    Parameters
    ----------
    kind : {"lstm", "gru"}
    hidden : int
        Hidden-state size.
    """

    def __init__(self, kind="lstm", hidden=32):
        super().__init__()
        if kind not in ("lstm", "gru"):
            raise ValueError("kind must be 'lstm' or 'gru'")
        self.kind, self.hidden = kind, hidden
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[kind](3, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, h=None):
        """Run over a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, time, 3), from :func:`features`.
        h : optional
            Recurrent state from a previous call (for stateful/chunked operation).

        Returns
        -------
        mu, logvar : torch.Tensor
            Shape (batch, time). ``mu`` is offset by 0.5 so an untrained net starts mid-range;
            ``logvar`` is clamped to [-14, 2] for numerical safety.
        h
            Final recurrent state.
        """
        y, h = self.rnn(x, h)
        out = self.head(y)
        return out[..., 0] + 0.5, out[..., 1].clamp(-14, 2), h


def augment(run, rng, p_clean=0.25):
    """Randomly corrupted (v, i, T) at 10 Hz; with probability ``p_clean`` the clean signals."""
    n = len(run.t)
    if rng.random() < p_clean:
        return run.v, run.i, run.T
    v = run.v + rng.uniform(0, 0.01) * rng.standard_normal(n)
    i = ((1 + rng.uniform(-0.01, 0.01)) * run.i + rng.uniform(-0.05, 0.05)
         + rng.uniform(0, 0.1) * rng.standard_normal(n))
    T = run.T + rng.uniform(0, 0.5) * rng.standard_normal(n)
    return v, i, T


def make_batch(runs, rng, max_start_frac=0.6):
    """Padded training batch from augmented runs with random start points.

    Returns
    -------
    X, Y, M : torch.Tensor
        Inputs (batch, time, 3), targets (batch, time) and a 0/1 padding mask (batch, time).
    """
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
    """beta-NLL loss and plain MSE over the unmasked samples.

    Plain Gaussian NLL (beta = 0) lets the network "explain away" hard regions with a large variance and
    stop fitting the mean there; weighting by var^beta (no gradient through the weight) keeps the mean
    accurate while the variance head still learns.

    Parameters
    ----------
    mu, logvar, y, m : torch.Tensor
        Predictions, targets and mask, shape (batch, time).
    warm : int
        Number of initial steps per sequence excluded from the loss (the network needs some data before
        it can know the SoC).
    beta : float, optional

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        beta-NLL and MSE.
    """
    m = m.clone()
    m[:, :warm] = 0
    nll = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp()) * logvar.detach().exp() ** beta
    return (nll * m).sum() / m.sum(), (((y - mu) ** 2) * m).sum() / m.sum()


@torch.no_grad()
def predict(model, v, i, T):
    """Run a trained network statefully over a full 10 Hz record.

    Returns
    -------
    mu, sigma : numpy.ndarray
        1 Hz mean SoC (fraction) and standard deviation.
    """
    x = torch.from_numpy(features(to_1hz(v), to_1hz(i), to_1hz(T)))[None]
    mu, logvar, _ = model(x)
    return mu[0].numpy(), (0.5 * logvar[0]).exp().numpy()


def estimate(nets, v, i, T):
    """SoC estimate on the 10 Hz grid from one network or a deep ensemble.

    Parameters
    ----------
    nets : list of SocNet
    v, i, T : numpy.ndarray
        10 Hz measurements.

    Returns
    -------
    mu, sigma : numpy.ndarray
        10 Hz (causally held) mean and standard deviation. For an ensemble the variance is the mean of the
        members' variances plus the variance of their means.
    """
    outs = [predict(net, v, i, T) for net in nets]
    mu = np.mean([o[0] for o in outs], 0)
    var = np.mean([o[1] ** 2 for o in outs], 0) + np.var([o[0] for o in outs], 0)
    return to_10hz(mu, len(v)), to_10hz(np.sqrt(var), len(v))


def load_model(tag):
    """Load ``results/ml_<tag>.pt``; ``tag`` looks like ``lstm32_c500_s0`` or ``gru32_c500_s1``."""
    kind = "gru" if tag.startswith("gru") else "lstm"
    net = SocNet(kind, int(tag[len(kind):].split("_")[0]))
    net.load_state_dict(torch.load(RESULTS / f"ml_{tag}.pt"))
    return net.eval()


def train(kind="lstm", hidden=32, chunk=500, epochs=600, warm=60, seed=0, lr=1e-3, verbose=True):
    """Train one network on the training cycles, selecting the epoch with the best validation RMSE.

    Parameters
    ----------
    kind : {"lstm", "gru"}
    hidden : int
        Hidden-state size.
    chunk : int
        Truncated-BPTT window [steps = s]: gradients flow back at most this far.
    epochs : int
    warm : int
        Steps at the start of each sequence excluded from the loss.
    seed : int
        Seeds both PyTorch and the augmentation/start-point generator.
    lr : float
        Initial Adam learning rate (cosine-annealed to zero).
    verbose : bool

    Returns
    -------
    model : SocNet
        Weights from the best validation epoch (among the NLL-phase epochs).
    info : dict
        Settings, best validation RMSE [% SoC], validation history and training time [s].
    """
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
        tot, nchunks = 0.0, 0
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
            nchunks += 1
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            model.eval()
            e = [np.sqrt(np.mean((predict(model, r.v, r.i, r.T)[0] - to_1hz(r.soc)) ** 2)) * 100
                 for r in val_runs]
            val = float(np.mean(e))
            hist.append(dict(epoch=ep + 1, train_rmse=float(np.sqrt(tot / nchunks) * 100), val_rmse=val))
            if val < best and use_nll:
                best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  ep {ep+1:4d}  train {hist[-1]['train_rmse']:5.2f}%  val {val:5.2f}%  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, dict(kind=kind, hidden=hidden, chunk=chunk, epochs=epochs, seed=seed, lr=lr,
                       best_val=best, history=hist, train_time_s=time.time() - t0)
