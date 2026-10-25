"""Computational cost: FLOP counts, arithmetic intensity, memory, measured runtime and profiles.

* FLOPs per update are counted analytically from the kernels (adds, multiplies, divides; transcendental
  calls such as exp/tanh/sigmoid are counted separately because on a microcontroller each costs roughly
  20-100 cycles).
* Arithmetic intensity = FLOPs per byte of parameters + state that must be read per update. Low intensity
  means memory-bound.
* Runtime per step is measured (best of several repetitions) for compiled scalar loops (numba), which is
  the fair proxy for embedded C, and for the unoptimised versions (NumPy reference EKF, PyTorch per-step
  calls) to quantify the optimisation.
* ``cProfile`` of the end-to-end ``estimate_soc`` call shows where wall time goes.

    python scripts/perf.py          writes results/perf.json and results/profile_*.txt
"""
import cProfile
import io
import json
import pstats
import time

import numpy as np
import torch
from numba import njit

from socest.api import estimate_soc, load_ekf_config
from socest.data import DT, RESULTS, load_cycle
from socest.ekf import ekf, ekf_reference
from socest.ml import SocNet
from socest.ocv import OCV

MCU_HZ = 168e6          # e.g. an ARM Cortex-M4F at 168 MHz
MCU_FLOP_CYCLES = 1.0   # single-precision FPU: ~1 cycle per add/mul (optimistic; no stalls)
MCU_TRANS_CYCLES = 60   # exp/tanh via library call, order of magnitude


def ekf_cost(n_rc):
    """Itemised operation count of one :func:`socest.ekf.ekf` step (dense nx x nx algebra)."""
    nx = 1 + n_rc
    lookup_param = 15          # knot interpolation + log-linear temperature term (+ 1 exp)
    lookup_ocv = 37            # 3 table rows: value and slope
    upd = (3 + lookup_ocv + lookup_param + 4 + n_rc        # Arrhenius variable, y and H[0]
           + 2 * nx * nx + 2 * nx + 1                      # P H^T, innovation variance, innovation
           + 3 * nx                                        # gain, state correction
           + 2 * nx * nx + 4 * nx ** 3 + 3 * nx * nx + 1)  # Joseph-form covariance update, sqrt
    pred = 4 + n_rc * (2 * lookup_param + 18) + 4 * nx ** 3 + nx
    params = (1 + 2 * n_rc) * 9 + 3 * 2            # theta + the two OCV table columns touched
    state = nx + nx * nx
    return dict(flops=upd + pred, transcendental=1 + 3 * n_rc + 1, bytes_read=8 * (params + state),
                params_total=(1 + 2 * n_rc) * 9 + 3 * 1101, state_floats=state)


def rnn_cost(kind, h, n_in=3):
    """Operation count of one recurrent step + output head."""
    g = {"lstm": 4, "gru": 3}[kind]
    macs = g * h * (n_in + h) + 2 * h
    elem = {"lstm": 5 * h, "gru": 6 * h}[kind]
    params = g * h * (n_in + h + 2) + 2 * (h + 1)
    state = {"lstm": 2 * h, "gru": h}[kind]
    return dict(flops=2 * macs + 2 * g * h + elem, transcendental={"lstm": 5 * h, "gru": 3 * h}[kind] + 1,
                bytes_read=4 * (params + state), params_total=params, state_floats=state)


@njit(cache=True)
def lstm_numba(Wi, Wh, b, Wo, bo, X):
    """Scalar-loop LSTM step (PyTorch gate order i, f, g, o), as it would be written in embedded C."""
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


def us_per_step(fn, n, reps=5):
    """Best-of-``reps`` wall time per step [us], after one warm-up call (JIT compilation)."""
    fn()
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best / n * 1e6


def mcu_us(c):
    """Projected time per update on the reference microcontroller [us] (order-of-magnitude estimate)."""
    return (c["flops"] * MCU_FLOP_CYCLES + c["transcendental"] * MCU_TRANS_CYCLES) / MCU_HZ * 1e6


def profile(label, fn):
    """Run fn under cProfile and save the top 15 functions by cumulative time."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    (RESULTS / f"profile_{label}.txt").write_text(s.getvalue())


def main():
    """Measure and count everything; write results/perf.json."""
    torch.set_num_threads(1)
    ocv = OCV()
    run = load_cycle(25, "UDDS")
    n = len(run.t)
    res = {}
    _, q, _ = load_ekf_config(2)
    for n_rc in (1, 2, 3):
        f = RESULTS / f"ecm_theta_{n_rc}rc.npy"
        if not f.exists():
            continue
        th = np.load(f)
        c = ekf_cost(n_rc)
        args = (th, run.v, run.i, run.T, DT, ocv.table, ocv.s0, ocv.ds, 1.0, 0.04, *q)
        e = dict(c, rate_hz=10, us_per_step=us_per_step(lambda: ekf(*args), n))
        if n_rc == 2:
            m = 20000   # the reference is slow: time a shorter slice
            short = (th, run.v[:m], run.i[:m], run.T[:m], DT, ocv.table, ocv.s0, ocv.ds, 1.0, 0.04, *q)
            e["numpy_reference_us_per_step"] = us_per_step(lambda: ekf_reference(*short), m, reps=2)
        res[f"ekf_{n_rc}rc"] = e

    X = np.random.default_rng(0).standard_normal((n // 10, 3))
    for kind, h in [("lstm", 16), ("lstm", 32), ("lstm", 64), ("gru", 32)]:
        c = rnn_cost(kind, h)
        e = dict(c, rate_hz=1)
        net = SocNet(kind, h).eval()
        if kind == "lstm":
            r = net.rnn
            Wi, Wh = r.weight_ih_l0.detach().double().numpy(), r.weight_hh_l0.detach().double().numpy()
            b = (r.bias_ih_l0 + r.bias_hh_l0).detach().double().numpy()
            Wo, bo = net.head.weight.detach().double().numpy()[:1], net.head.bias.detach().double().numpy()[:1]
            e["us_per_step"] = us_per_step(lambda: lstm_numba(Wi, Wh, b, Wo, bo, X), len(X))
            with torch.no_grad():
                ref = net(torch.from_numpy(X[:500].astype(np.float32))[None])[0][0].numpy()
            e["numba_vs_torch_maxdiff"] = float(np.max(np.abs(lstm_numba(Wi, Wh, b, Wo, bo, X[:500]) - ref)))
        xt = torch.from_numpy(X[:2000].astype(np.float32))[None]

        def torch_steps():
            """One PyTorch call per time step (as a naive deployment would)."""
            hs = None
            with torch.no_grad():
                for t in range(xt.shape[1]):
                    _, _, hs = net(xt[:, t:t + 1], hs)
        e["torch_per_step_us"] = us_per_step(torch_steps, xt.shape[1], reps=2)
        with torch.no_grad():
            e["torch_whole_sequence_us_per_step"] = us_per_step(lambda: net(xt), xt.shape[1], reps=3)
        res[f"{kind}{h}"] = e

    for k, e in res.items():
        e["flops_per_s"] = e["flops"] * e["rate_hz"]
        e["arith_intensity"] = e["flops"] / e["bytes_read"]
        e["mcu_us_per_step"] = mcu_us(e)
        e["mcu_cpu_load_pct"] = e["mcu_us_per_step"] * e["rate_hz"] / 1e4

    import socest.api as api
    api._ocv_cache.clear()
    profile("ekf_pipeline", lambda: estimate_soc(run.v, run.i, run.T, method="ekf", soc0=1.0))
    if (RESULTS / "ml_selected.json").exists():
        profile("ml_pipeline", lambda: estimate_soc(run.v, run.i, run.T, method="ml"))

    (RESULTS / "perf.json").write_text(json.dumps(dict(
        mcu=dict(hz=MCU_HZ, flop_cycles=MCU_FLOP_CYCLES, trans_cycles=MCU_TRANS_CYCLES), models=res), indent=1))
    print(f"{'model':9s} {'Hz':>3s} {'FLOP/step':>9s} {'trans':>5s} {'FLOP/s':>8s} {'params':>6s} "
          f"{'FLOP/B':>6s} {'us/step':>8s} {'MCU us':>7s} {'MCU load':>8s}")
    for k, e in res.items():
        print(f"{k:9s} {e['rate_hz']:>3d} {e['flops']:>9d} {e['transcendental']:>5d} {e['flops_per_s']:>8d} "
              f"{e['params_total']:>6d} {e['arith_intensity']:>6.2f} {e.get('us_per_step', float('nan')):>8.2f} "
              f"{e['mcu_us_per_step']:>7.1f} {e['mcu_cpu_load_pct']:>7.3f}%")
    e = res["ekf_2rc"]
    print(f"EKF 2RC: numba {e['us_per_step']:.2f} us/step vs NumPy reference "
          f"{e['numpy_reference_us_per_step']:.1f} us/step -> {e['numpy_reference_us_per_step']/e['us_per_step']:.0f}x")
    for k in ("lstm32",):
        e = res[k]
        print(f"{k}: numba {e['us_per_step']:.2f} us/step, PyTorch per-step call {e['torch_per_step_us']:.1f} us, "
              f"PyTorch whole sequence {e['torch_whole_sequence_us_per_step']:.2f} us/step; "
              f"numba-vs-torch max diff {e['numba_vs_torch_maxdiff']:.1e}")


if __name__ == "__main__":
    main()
