"""Metrics on hand-computable cases, sensor scenarios, and the top-level API."""
import numpy as np
import pytest

from socest.data import DT, Q_REF, Run
from socest.metrics import soc_metrics
from socest.sensors import SCENARIOS, corrupt


def make_run(n=20000, soc0=1.0, current=-1.0):
    t = np.arange(n) * DT
    ah = current * t / 3600 + (soc0 - 1) * Q_REF
    return Run("synthetic", 25, t, np.full(n, 3.7), np.full(n, current), np.full(n, 25.0), ah)


def test_metrics_constant_offset():
    r = make_run(soc0=0.9)    # start below full so +2 % stays inside [0, 1]
    m = soc_metrics(r.soc + 0.02, r)
    assert m["rmse"] == pytest.approx(2.0) and m["maxabs"] == pytest.approx(2.0)
    assert m["final"] == pytest.approx(2.0) and m["frac_outside_01"] == 0.0


def test_metrics_settled_excludes_first_10_minutes():
    r = make_run()
    est = r.soc.copy()
    est[r.t < 300] += 0.2                      # large error only during convergence
    m = soc_metrics(est, r)
    assert m["maxabs"] == pytest.approx(20.0) and m["maxabs_settled"] == pytest.approx(0.0)


def test_coverage_and_charge_bookkeeping():
    r = make_run()
    m = soc_metrics(r.soc + 0.01, r, i_meas=r.i, sig=np.full(len(r.t), 0.005))
    assert m["coverage_3sig"] == 1.0                     # 1 % error inside 3 x 0.5 %
    assert abs(m["unexplained_dsoc"]) < 1e-6              # perfectly charge-consistent estimate


def test_physical_range_check():
    r = make_run()
    m = soc_metrics(np.linspace(-0.1, 1.1, len(r.t)), r)
    assert 0.15 < m["frac_outside_01"] < 0.18


def test_scenarios_are_reproducible_and_leave_truth_untouched():
    r = make_run()
    truth = r.soc.copy()
    a = corrupt(r, SCENARIOS["realistic"], seed=3)
    b = corrupt(r, SCENARIOS["realistic"], seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a[:3], b[:3]))
    assert np.array_equal(r.soc, truth)
    assert a[3] == pytest.approx(0.9)                      # 10-point initial-SoC error


def test_api_ekf_tracks_synthetic_discharge(synthetic_ocv, theta2, monkeypatch):
    """estimate_soc end to end on a model-generated discharge, starting from an OCV-based guess."""
    from socest import api
    from socest.ecm import simulate
    from socest.ekf import coulomb_count
    monkeypatch.setattr(api, "_ocv", lambda: synthetic_ocv)
    monkeypatch.setattr(api, "load_ekf_config", lambda n_rc=2: (theta2, (1e-8, 1e-7, 0.01), 1.0))
    n = 36000
    i = np.repeat(np.random.default_rng(0).uniform(-6, 2, n // 100), 100)
    T = np.full(n, 25.0)
    s_true = coulomb_count(i, DT, 0.95)
    v = simulate(theta2, s_true, T, i, DT, synthetic_ocv.table, synthetic_ocv.s0, synthetic_ocv.ds)
    out = api.estimate_soc(v, i, T, method="ekf")
    late = n // 4
    assert np.sqrt(np.mean((out["soc"][late:] - s_true[late:]) ** 2)) < 0.01
    assert np.all((out["soc"] >= 0) & (out["soc"] <= 1))


def test_api_rejects_unknown_method(synthetic_ocv, monkeypatch):
    from socest import api
    monkeypatch.setattr(api, "_ocv", lambda: synthetic_ocv)
    with pytest.raises(ValueError, match="unknown method"):
        api.estimate_soc([3.7, 3.7], [0, 0], [25, 25], method="magic")


def test_api_ml_requires_10hz():
    from socest import api
    with pytest.raises(ValueError, match="10 Hz"):
        api.estimate_soc(np.full(10, 3.7), np.zeros(10), np.full(10, 25.0), dt=1.0, method="ml")
