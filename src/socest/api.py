"""Top-level interface: estimate SoC from measured voltage, current and temperature.

Example
-------
>>> from socest.api import estimate_soc
>>> out = estimate_soc(v, i, T, dt=0.1, method="ekf")      # doctest: +SKIP
>>> out["soc"], out["sigma"]                                 # doctest: +SKIP

Inputs
------
``v`` terminal voltage [V], ``i`` current [A, discharge negative], ``T`` cell temperature [degC], equal-length
1-D arrays on a uniform grid of period ``dt`` [s]. Validated by :func:`socest.validation.validate_inputs`.

Outputs
-------
A dict with ``soc`` (fraction, clipped to the physical range [0, 1]), ``soc_raw`` (unclipped),
``sigma`` (calibrated 1-sigma uncertainty, fraction) and ``method``.

Fitted parameters are read from ``results/`` (produced by ``scripts/run_all.py``).
"""
import json

import numpy as np

from .data import DT, RESULTS
from .ekf import coulomb_count, ekf
from .ocv import OCV
from .validation import validate_inputs

P0_S = 0.2 ** 2
"""Prior SoC variance used by the EKF (the initial guess may be ~20 points off)."""

_ocv_cache = {}


OCV_FILE = RESULTS / "ocv_model.npz"
"""Fitted OCV model written by ``scripts/fit_ecm.py``. cProfile showed 93 % of ``estimate_soc`` time
(1.05 of 1.12 s on a 6.2 h cycle) was spent rebuilding OCV from the raw .mat files; loading this file
instead brings the call to 0.08 s, of which 0.07 s is the EKF itself (plus a one-off ~0.3 s for numba
to load its compiled code in a fresh process). It also removes the need for the raw data."""


def _ocv():
    if "ocv" not in _ocv_cache:
        _ocv_cache["ocv"] = OCV.load(OCV_FILE) if OCV_FILE.exists() else OCV()
    return _ocv_cache["ocv"]


def selected_n_rc():
    """ECM order chosen on validation (``results/ekf_selected.json``), default 2 if not yet selected."""
    p = RESULTS / "ekf_selected.json"
    return json.loads(p.read_text())["n_rc"] if p.exists() else 2


def load_ekf_config(n_rc=None):
    """Fitted ECM parameters and tuned EKF settings.

    Parameters
    ----------
    n_rc : int, optional
        ECM order; defaults to the validation-selected one.

    Returns
    -------
    theta : numpy.ndarray
    q : tuple of float
        (q_s, q_v, r) tuned on the validation cycle at 10 Hz.
    alpha : float
        Sigma calibration factor.
    """
    n_rc = n_rc or selected_n_rc()
    theta = np.load(RESULTS / f"ecm_theta_{n_rc}rc.npy")
    cfg = json.loads((RESULTS / f"ekf_{n_rc}rc_eval.json").read_text())
    return theta, tuple(cfg["q"]), cfg["alpha"]


def estimate_soc(v, i, T, dt=DT, method="ekf", soc0=None, n_rc=None, ml_tags=None):
    """Estimate state of charge.

    Parameters
    ----------
    v, i, T : array_like
        Voltage [V], current [A, discharge negative], cell temperature [degC].
    dt : float, optional
        Sample period [s]. The EKF accepts any dt (its discretisation is exact; process noise is scaled
        from the 10 Hz tuning); the ML model requires 0.1 s.
    method : {"ekf", "coulomb", "ml"}
    soc0 : float, optional
        Initial SoC. If omitted, EKF and Coulomb counting initialise from the OCV of the first sample,
        which assumes the cell has been resting. Ignored by the ML model.
    n_rc : int, optional
        RC pairs of the ECM used by the EKF; defaults to the order selected on validation.
    ml_tags : list of str, optional
        Trained networks to use (an ensemble if several). Defaults to the selection saved by
        ``scripts/run_all.py`` in ``results/ml_selected.json``.

    Returns
    -------
    dict
        ``soc``, ``soc_raw``, ``sigma`` (numpy arrays) and ``method``.

    Raises
    ------
    ValueError
        Invalid inputs (see :mod:`socest.validation`), unknown method, or an unsupported ``dt`` for ML.
    """
    v, i, T = validate_inputs(v, i, T, dt=dt)
    if method not in ("ekf", "coulomb", "ml"):
        raise ValueError(f"unknown method {method!r}")
    if method == "ml":
        from .ml import estimate, load_model   # torch is only imported when needed
        if abs(dt - DT) > 1e-9:
            raise ValueError("the ML model was trained on 10 Hz data; resample to dt = 0.1 s")
        if ml_tags is None:
            ml_tags = json.loads((RESULTS / "ml_selected.json").read_text())["tags"]
            alpha = json.loads((RESULTS / "ml_selected.json").read_text())["alpha"]
        else:
            alpha = 1.0
        mu, sig = estimate([load_model(t) for t in ml_tags], v, i, T)
        return dict(soc=np.clip(mu, 0, 1), soc_raw=mu, sigma=alpha * sig, method="ml")
    if soc0 is None:
        soc0 = _ocv().inverse(v[0], T[0])
    if method == "coulomb":
        s = coulomb_count(i, dt, soc0)
        return dict(soc=np.clip(s, 0, 1), soc_raw=s, sigma=np.full_like(s, np.nan), method="coulomb")
    theta, (q_s, q_v, r), alpha = load_ekf_config(n_rc)
    scale = dt / DT     # process noise was tuned per 0.1 s step; keep it per unit time
    ocv = _ocv()
    s, sig, _ = ekf(theta, v, i, T, dt, ocv.table, ocv.s0, ocv.ds, soc0, P0_S, q_s * scale, q_v * scale, r)
    return dict(soc=np.clip(s, 0, 1), soc_raw=s, sigma=alpha * sig, method="ekf")
