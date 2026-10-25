"""OCV model: structural properties (synthetic) and accuracy against relaxed HPPC rests (real data)."""
import numpy as np
import pytest

from socest.ocv import hppc_rest_points


def test_monotonic_and_invertible(synthetic_ocv):
    s = np.linspace(0, 1, 501)
    for T in (-20, 0, 25):
        v = synthetic_ocv(s, T)
        assert np.all(np.diff(v) > 0)
        for target in (0.1, 0.5, 0.9):
            assert synthetic_ocv.inverse(synthetic_ocv(target, T), T) == pytest.approx(target, abs=1e-3)


def test_slope_matches_finite_difference(synthetic_ocv):
    s = np.linspace(0.05, 0.95, 50) + 0.0005    # mid-cell, away from grid nodes
    fd = (synthetic_ocv(s + 1e-5, 10) - synthetic_ocv(s - 1e-5, 10)) / 2e-5
    assert np.allclose(synthetic_ocv.dv_ds(s, 10), fd, rtol=1e-6)


def test_inverse_clips_to_physical_range(synthetic_ocv):
    assert synthetic_ocv.inverse(5.0, 25) == 1.0
    assert synthetic_ocv.inverse(1.0, 25) == 0.0


@pytest.mark.parametrize("temp, limit_mV", [(25, 6), (0, 8), (-20, 10), (10, 10), (-10, 10)])
def test_matches_relaxed_rests(real_ocv, temp, limit_mV):
    """RMSE vs HPPC rest voltages (SoC 0.1-0.95). 10 and -10 degC were not used to build the model."""
    p = hppc_rest_points(temp)
    m = (p[:, 0] >= 0.1) & (p[:, 0] <= 0.95)
    rmse = np.sqrt(np.mean((p[m, 1] - real_ocv(p[m, 0], temp)) ** 2)) * 1000
    assert rmse < limit_mV
