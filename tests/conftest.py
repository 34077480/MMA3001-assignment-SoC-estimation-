"""Shared fixtures.

Most numerical tests use a *synthetic* OCV curve so they test the code, not the data, and run without the
198 MB dataset. Tests that need the raw data or fitted parameters are skipped when those are absent.
"""
import numpy as np
import pytest

from socest.data import RAW, RESULTS
from socest.ocv import K_KNOTS, OCV, S_GRID

needs_data = pytest.mark.skipif(not RAW.exists(), reason="raw dataset not downloaded (see README)")


@pytest.fixture(scope="session")
def synthetic_ocv():
    """Smooth, strictly increasing, temperature-dependent OCV resembling an NCA cell."""
    ocv25 = 3.3 + 0.75 * S_GRID + 0.12 * np.tanh(8 * (S_GRID - 0.05)) + 0.05 * S_GRID ** 3
    n = len(K_KNOTS)
    tab = dict(s=S_GRID, ocv25=ocv25, k_knots=K_KNOTS, k1=np.full(n, 0.8e-3), k2=np.full(n, -0.01e-3))
    return OCV(tab)


@pytest.fixture(scope="session")
def real_ocv():
    if not RAW.exists():
        pytest.skip("raw dataset not downloaded")
    return OCV()


@pytest.fixture(scope="session")
def theta2():
    """Fitted 2-RC parameters (committed in results/)."""
    p = RESULTS / "ecm_theta_2rc.npy"
    if not p.exists():
        pytest.skip("fitted parameters not found; run scripts/run_all.py")
    return np.load(p)
