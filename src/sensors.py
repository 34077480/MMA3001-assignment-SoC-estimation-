"""Measurement-corruption scenarios applied identically to every estimator.

The lab data are near-perfect (the ground truth IS Coulomb counting on the cycler's precision shunt), so
Coulomb counting on clean data is trivially exact. A production BMS has a cheaper current sensor and does
not know its starting SoC, so robustness is tested by corrupting what the estimator sees:

    i_meas = (1 + gain) i + bias + N(0, sig_i)     v_meas = v + N(0, sig_v)     T_meas = T + N(0, sig_T)

Ground truth is untouched.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    soc0_err: float = 0.0   # initial SoC guess = true SoC0 + soc0_err
    i_bias: float = 0.0     # A
    i_gain: float = 0.0     # fractional
    sig_i: float = 0.0      # A, white noise per sample
    sig_v: float = 0.0      # V
    sig_T: float = 0.0      # degC


SCENARIOS = {s.name: s for s in [
    Scenario("clean"),
    Scenario("soc0_-20%", soc0_err=-0.20),
    Scenario("i_bias_+50mA", i_bias=0.05),
    Scenario("i_bias_-50mA", i_bias=-0.05),
    Scenario("noise", sig_i=0.1, sig_v=0.01, sig_T=0.5),
    # a plausible low-cost BMS: unknown start, small offset and gain error, noisy channels
    Scenario("realistic", soc0_err=-0.10, i_bias=0.03, i_gain=0.01, sig_i=0.05, sig_v=0.005, sig_T=0.5),
]}


def corrupt(run, sc, seed=0):
    """Return (v, i, T, soc0_guess) as the estimator would see them."""
    rng = np.random.default_rng(seed)
    n = len(run.t)
    v = run.v + sc.sig_v * rng.standard_normal(n)
    i = (1 + sc.i_gain) * run.i + sc.i_bias + sc.sig_i * rng.standard_normal(n)
    T = run.T + sc.sig_T * rng.standard_normal(n)
    soc0 = float(np.clip(run.soc[0] + sc.soc0_err, 0.0, 1.0))
    return v, i, T, soc0
