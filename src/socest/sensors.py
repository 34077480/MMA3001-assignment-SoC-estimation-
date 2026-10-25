"""Measurement-corruption scenarios, applied identically to every estimator.

Why this exists: the lab ground truth *is* Coulomb counting on the cycler's precision shunt, so Coulomb
counting on the clean lab data is trivially exact. A production BMS has a cheaper current sensor and does
not know its starting SoC, so robustness is tested by corrupting what the estimator sees::

    i_meas = (1 + gain) i + bias + N(0, sig_i)
    v_meas = v + N(0, sig_v)
    T_meas = T + N(0, sig_T)

The ground truth is never modified. Noise is white per 10 Hz sample and seeded, so every estimator sees
exactly the same corrupted signals.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """One sensor-corruption scenario.

    Attributes
    ----------
    name : str
    soc0_err : float
        Initial SoC guess = true initial SoC + ``soc0_err`` (fraction), clipped to [0, 1].
    i_bias : float
        Current-sensor offset [A].
    i_gain : float
        Current-sensor gain error (fraction).
    sig_i, sig_v, sig_T : float
        Standard deviations of white noise on current [A], voltage [V] and temperature [degC].
    """
    name: str
    soc0_err: float = 0.0
    i_bias: float = 0.0
    i_gain: float = 0.0
    sig_i: float = 0.0
    sig_v: float = 0.0
    sig_T: float = 0.0


SCENARIOS = {s.name: s for s in [
    Scenario("clean"),
    Scenario("soc0_-20%", soc0_err=-0.20),
    Scenario("i_bias_+50mA", i_bias=0.05),
    Scenario("i_bias_-50mA", i_bias=-0.05),
    Scenario("noise", sig_i=0.1, sig_v=0.01, sig_T=0.5),
    # a plausible low-cost BMS: unknown start, small offset and gain error, noisy channels
    Scenario("realistic", soc0_err=-0.10, i_bias=0.03, i_gain=0.01, sig_i=0.05, sig_v=0.005, sig_T=0.5),
]}
"""All evaluation scenarios, by name."""

TUNE_SCENARIOS = ["clean", "soc0_-20%", "i_bias_+50mA", "realistic"]
"""Scenarios used (on the validation cycle only) to tune the EKF and to calibrate uncertainties."""


def corrupt(run, sc, seed=0):
    """Return the signals an estimator would see under a scenario.

    Parameters
    ----------
    run : socest.data.Run
    sc : Scenario
    seed : int, optional
        Seed for the noise generator.

    Returns
    -------
    v, i, T : numpy.ndarray
        Corrupted voltage [V], current [A] and temperature [degC].
    soc0 : float
        Initial SoC guess handed to estimators that need one.
    """
    rng = np.random.default_rng(seed)
    n = len(run.t)
    v = run.v + sc.sig_v * rng.standard_normal(n)
    i = (1 + sc.i_gain) * run.i + sc.i_bias + sc.sig_i * rng.standard_normal(n)
    T = run.T + sc.sig_T * rng.standard_normal(n)
    soc0 = float(np.clip(run.soc[0] + sc.soc0_err, 0.0, 1.0))
    return v, i, T, soc0
