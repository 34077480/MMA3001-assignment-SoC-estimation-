"""Dataset loading and the ground-truth definition (needs the raw dataset)."""
import numpy as np
import pytest

from conftest import needs_data
from socest.data import CYCLES, DT, Q_REF, TEMP_DIRS, TEST_CYCLES, TRAIN_CYCLES, VAL_CYCLES, find_cycle_file, load_cycle

pytestmark = needs_data


def test_split_is_disjoint_and_complete():
    tr, va, te = set(TRAIN_CYCLES), set(VAL_CYCLES), set(TEST_CYCLES)
    assert not (tr & va or tr & te or va & te)
    assert tr | va | te == set(CYCLES)


@pytest.mark.parametrize("temp", list(TEMP_DIRS))
def test_file_matching_is_unambiguous(temp):
    """Regression test for the bug where '0degC' also matched 'n10degC' files."""
    for c in CYCLES:
        f = find_cycle_file(temp, c)
        prefix = f"{'n' if temp < 0 else ''}{abs(temp)}degC_"
        assert f.name.split(" ", 1)[1].startswith(prefix)


@pytest.mark.parametrize("temp, cycle", [(25, "UDDS"), (-10, "US06"), (0, "LA92")])
def test_resampled_run(temp, cycle):
    r = load_cycle(temp, cycle)
    assert np.allclose(np.diff(r.t), DT)
    assert r.soc[0] == pytest.approx(1.0, abs=0.01)          # every cycle starts full
    assert 2.4 < r.v.min() and r.v.max() < 4.25
    assert abs(r.T[0] - temp) < 2.0                           # leading warm rest has been trimmed


@pytest.mark.parametrize("temp, cycle", [(25, "UDDS"), (-20, "US06")])
def test_ground_truth_is_charge_conserving(temp, cycle):
    """Our own integration of the logged current must reproduce the cycler's Ah counter (< 0.2 % SoC)."""
    r = load_cycle(temp, cycle)
    cc = np.concatenate([[0], np.cumsum(r.i[:-1] * DT)]) / 3600
    assert np.max(np.abs(cc - (r.ah - r.ah[0]))) / Q_REF < 0.002


def test_unknown_temperature():
    with pytest.raises(KeyError):
        find_cycle_file(15, "UDDS")
