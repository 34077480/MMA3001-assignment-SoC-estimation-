"""Verify the RC discretisation against the analytical step response.

For a current step I applied at t = 0 to an RC pair starting relaxed:
    v(t) = R I (1 - exp(-t / tau))
Exact zero-order hold must reproduce this to rounding error for ANY dt; forward Euler is first order and
unstable for dt > 2 tau; Tustin (bilinear) is second order. Also checks ecm.simulate end to end against
the analytical 2-RC terminal voltage.

Run with pytest, or ``python tests/test_integrator.py`` to also write
``results/figures/integrator_verification.png``.
"""
import numpy as np

from socest.data import RESULTS
from socest.ecm import N_COEF, N_KNOTS, simulate

R, TAU, I = 0.015, 20.0, -5.0
T_END = 200.0


def step(method, dt, t_end=T_END):
    n = int(round(t_end / dt))
    v = np.zeros(n + 1)
    a = {"zoh": np.exp(-dt / TAU), "euler": 1 - dt / TAU, "tustin": (1 - dt / (2 * TAU)) / (1 + dt / (2 * TAU))}[method]
    b = {"zoh": R * (1 - a), "euler": R * dt / TAU, "tustin": R * (1 - a)}[method]
    for k in range(n):
        v[k + 1] = a * v[k] + b * I
    t = np.arange(n + 1) * dt
    return t, v


def exact(t):
    return R * I * (1 - np.exp(-t / TAU))


def max_err(method, dt):
    t, v = step(method, dt)
    return np.max(np.abs(v - exact(t)))


def test_zoh_exact():
    for dt in [0.01, 0.1, 1.0, 10.0, 50.0]:   # up to 2.5 tau
        assert max_err("zoh", dt) < 1e-12, dt


def test_convergence_orders():
    dts = np.array([0.4, 0.2, 0.1, 0.05])
    for method, order in [("euler", 1), ("tustin", 2)]:
        e = np.array([max_err(method, dt) for dt in dts])
        p = np.polyfit(np.log(dts), np.log(e), 1)[0]
        assert abs(p - order) < 0.1, (method, p)


def test_euler_unstable_zoh_stable():
    _, ve = step("euler", 2.5 * TAU, t_end=40 * 2.5 * TAU)   # 40 steps: |1 - 2.5|^40 ~ 1e7
    _, vz = step("zoh", 2.5 * TAU, t_end=40 * 2.5 * TAU)
    assert np.abs(ve).max() > 1e3 * abs(R * I)
    assert np.abs(vz).max() <= abs(R * I) * (1 + 1e-12)


def test_simulate_matches_analytic_2rc(synthetic_ocv):
    """ecm.simulate with constant parameters vs V = OCV + R0 I + sum R_k I (1 - exp(-t/tau_k))."""
    ocv = synthetic_ocv
    R0, (R1, t1), (R2, t2) = 0.02, (0.01, 3.0), (0.02, 150.0)
    th = np.zeros((5, N_COEF))
    for row, val in enumerate([R0, R1, t1, R2, t2]):
        th[row, :N_KNOTS] = np.log(val)          # no SoC or temperature dependence
    dt, n = 0.1, 6000
    s, T, i = np.full(n, 0.5), np.full(n, 25.0), np.full(n, I)
    v = simulate(th, s, T, i, dt, ocv.table, ocv.s0, ocv.ds)
    t = np.arange(n) * dt
    va = ocv(0.5, 25) + R0 * I + R1 * I * (1 - np.exp(-t / t1)) + R2 * I * (1 - np.exp(-t / t2))
    assert np.max(np.abs(v - va)) < 1e-12


def figure():
    """Write results/figures/integrator_verification.png (step response and error vs step size)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(zoh="#2a78d6", euler="#eb6834", tustin="#1baf7a")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tt = np.linspace(0, T_END, 1000)
    ax[0].plot(tt, exact(tt) * 1000, color="#0b0b0b", lw=2, label="analytical")
    for m, st in [("zoh", "o"), ("euler", "s"), ("tustin", "^")]:
        t, v = step(m, 10.0)
        ax[0].plot(t, v * 1000, st + "--", color=col[m], ms=5, lw=1, label=f"{m}, dt = tau/2")
    ax[0].set(xlabel="t [s]", ylabel="v_RC [mV]",
              title=f"Step response: R = {R*1000:.0f} mOhm, tau = {TAU:.0f} s, I = {I} A")
    dts = np.logspace(-2, np.log10(30), 25)
    for m in ["zoh", "euler", "tustin"]:
        e = np.array([max_err(m, dt) for dt in dts])
        ax[1].loglog(dts / TAU, np.maximum(e, 1e-17) * 1000, "o-", color=col[m], ms=3, label=m)
    ax[1].axvline(2, color="#e34948", lw=0.8, ls=":")
    ax[1].text(1.9, 1e2, "Euler stability\nlimit dt = 2 tau ", color="#52514e", fontsize=8, ha="right")
    ax[1].axvline(0.1 / 3.0, color="#52514e", lw=0.8, ls=":")
    ax[1].text(0.036, 1e-10, " project dt /\n fastest tau", color="#52514e", fontsize=8)
    ax[1].set(xlabel="dt / tau", ylabel="max |error| [mV]", title="Discretisation error vs step size",
              ylim=(1e-15, 1e5))
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "integrator_verification.png", dpi=150)


if __name__ == "__main__":
    dts = np.array([0.4, 0.2, 0.1, 0.05])
    for m in ["euler", "tustin"]:
        e = [max_err(m, dt) for dt in dts]
        print(f"{m:6s} observed order {np.polyfit(np.log(dts), np.log(e), 1)[0]:.3f}")
    print(f"zoh max error at dt = tau/2: {max_err('zoh', 10.0):.2e} V")
    figure()
