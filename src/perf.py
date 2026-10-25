"""Compute cost per update: analytic FLOP counts, parameter memory, and measured runtime per timestep.

Both estimators are timed as compiled scalar loops (numba), which is the fair comparison for a
microcontroller target; the PyTorch per-step time is also reported to show framework overhead.
FLOPs count adds, multiplies, divides and compares-free arithmetic; transcendental calls (exp, tanh,
sigmoid) are counted separately because on an MCU they cost 10-50x a multiply.

    python src/perf.py
"""
import json
import time

import numpy as np
import torch
from numba import njit

from data import DT, ROOT, load_cycle
from ekf import ekf
from ml import SocNet
from ocv import OCV

OUT = ROOT / "results"


def ekf_cost(n_rc):
    """Operation count of one ekf.ekf step, itemised from the kernel (dense nx x nx matrix algebra)."""
    nx = 1 + n_rc
    param = 15                        # knot interpolation + log-linear temperature term, one exp
    upd = (3 + 37 + param + 4 + n_rc  # Arrhenius x, OCV + slope lookup, R0 lookup, y and H[0], sum v_m
           + 2 * nx * nx + 2 * nx + 1  # P H^T, innovation variance S, innovation
           + 3 * nx                    # gain and state correction
           + 2 * nx * nx + 4 * nx ** 3 + 3 * nx * nx + 1)   # Joseph-form covariance update, sqrt
    pred = 4 + n_rc * (2 * param + 2 + 4 + 7 + 5) + 4 * nx ** 3 + nx
    return dict(flops=upd + pred, transcendental=1 + 3 * n_rc + 1,   # R0, (R, tau, a) per pair, sqrt
                params=(1 + 2 * n_rc) * 9, table_floats=3 * 1101,
                state_floats=nx + nx * nx)


def rnn_cost(kind, h, n_in=3):
    gates = {"lstm": 4, "gru": 3}[kind]
    macs = gates * h * (n_in + h) + 2 * h            # gate matmuls + output head
    elem = {"lstm": 5 * h, "gru": 6 * h}[kind]        # gate combinations (c = f c + i g, h = o tanh c ...)
    trans = {"lstm": 5 * h, "gru": 3 * h}[kind]       # sigmoids/tanh, plus exp for sigma at the head
    params = gates * h * (n_in + h + 2) + 2 * (h + 1)
    return dict(flops=2 * macs + gates * 2 * h + elem, transcendental=trans + 1, params=params,
                state_floats={"lstm": 2 * h, "gru": h}[kind])


@njit(cache=True)
def lstm_numba(Wi, Wh, b, Wo, bo, X):
    """Plain scalar-loop LSTM (PyTorch gate order i, f, g, o), the way it would run on an MCU."""
    H = Wh.shape[1]
    h = np.zeros(H)
    c = np.zeros(H)
    out = np.empty(X.shape[0])
    z = np.empty(4 * H)
    for t in range(X.shape[0]):
        for r in range(4 * H):
            acc = b[r]
            for k in range(X.shape[1]):
                acc += Wi[r, k] * X[t, k]
            for k in range(H):
                acc += Wh[r, k] * h[k]
            z[r] = acc
        for k in range(H):
            ig = 1 / (1 + np.exp(-z[k]))
            fg = 1 / (1 + np.exp(-z[H + k]))
            gg = np.tanh(z[2 * H + k])
            og = 1 / (1 + np.exp(-z[3 * H + k]))
            c[k] = fg * c[k] + ig * gg
            h[k] = og * np.tanh(c[k])
        acc = bo[0]
        for k in range(H):
            acc += Wo[0, k] * h[k]
        out[t] = acc + 0.5
    return out


def time_per_step(fn, n_steps, reps=5):
    fn()                                   # JIT warm-up
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best / n_steps * 1e6           # microseconds per step


def main():
    ocv = OCV()
    run = load_cycle(25, "UDDS")
    n10 = len(run.t)
    res = {}
    q = json.loads((OUT / "ekf_2rc_eval.json").read_text())["q"]
    for n_rc in (1, 2, 3):
        f = OUT / f"ecm_theta_{n_rc}rc.npy"
        if not f.exists():
            continue
        th = np.load(f)
        us = time_per_step(lambda: ekf(th, run.v, run.i, run.T, DT, ocv.table, ocv.s0, ocv.ds,
                                       1.0, 0.04, *q), n10)
        c = ekf_cost(n_rc)
        res[f"ekf_{n_rc}rc"] = dict(c, us_per_step=us, rate_hz=10, flops_per_s=c["flops"] * 10,
                                    trans_per_s=c["transcendental"] * 10)
    X = np.random.default_rng(0).standard_normal((n10 // 10, 3))
    for kind, h in [("lstm", 16), ("lstm", 32), ("lstm", 64), ("gru", 32)]:
        c = rnn_cost(kind, h)
        entry = dict(c, rate_hz=1, flops_per_s=c["flops"], trans_per_s=c["transcendental"])
        net = SocNet(kind, h).eval()
        if kind == "lstm":
            r = net.rnn
            Wi, Wh = r.weight_ih_l0.detach().double().numpy(), r.weight_hh_l0.detach().double().numpy()
            b = (r.bias_ih_l0 + r.bias_hh_l0).detach().double().numpy()
            Wo, bo = net.head.weight.detach().double().numpy()[:1], net.head.bias.detach().double().numpy()[:1]
            entry["us_per_step"] = time_per_step(lambda: lstm_numba(Wi, Wh, b, Wo, bo, X), len(X))
            # check the scalar loop reproduces PyTorch exactly
            with torch.no_grad():
                ref = net(torch.from_numpy(X[:500].astype(np.float32))[None])[0][0].numpy()
            entry["numba_vs_torch_maxdiff"] = float(np.max(np.abs(lstm_numba(Wi, Wh, b, Wo, bo, X[:500]) - ref)))
        torch.set_num_threads(1)
        xt = torch.from_numpy(X[:2000].astype(np.float32))[None]

        def torch_steps():
            hstate = None
            with torch.no_grad():
                for t in range(xt.shape[1]):
                    _, _, hstate = net(xt[:, t:t + 1], hstate)
        entry["torch_us_per_step"] = time_per_step(torch_steps, xt.shape[1], reps=2)
        res[f"{kind}{h}"] = entry
    (OUT / "perf.json").write_text(json.dumps(res, indent=1))
    print(f"{'model':10s} {'rate':>5s} {'FLOP/step':>10s} {'trans/step':>10s} {'FLOP/s':>9s} {'params':>7s} "
          f"{'us/step':>8s} {'torch us':>9s}")
    for k, e in res.items():
        print(f"{k:10s} {e['rate_hz']:>4}Hz {e['flops']:>10d} {e['transcendental']:>10d} {e['flops_per_s']:>9d} "
              f"{e['params']:>7d} {e.get('us_per_step', float('nan')):>8.2f} {e.get('torch_us_per_step', float('nan')):>9.1f}")
        if "numba_vs_torch_maxdiff" in e:
            print(f"{'':10s} numba LSTM vs PyTorch max |diff| = {e['numba_vs_torch_maxdiff']:.1e}")


if __name__ == "__main__":
    main()
