"""Input checking: every row of the policy table in socest.validation has a test."""
import numpy as np
import pytest

from socest.validation import InputDomainWarning, validate_inputs


def good(n=100):
    t = np.arange(n) * 0.1
    return 3.7 + 0 * t, -2.0 + 0 * t, 25.0 + 0 * t


def test_clean_input_passes_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v, i, T = validate_inputs(*good())
    assert v.dtype == float and len(v) == 100


def test_length_mismatch():
    v, i, T = good()
    with pytest.raises(ValueError, match="equal length"):
        validate_inputs(v, i[:-1], T)


def test_not_1d():
    v, i, T = good()
    with pytest.raises(ValueError, match="1-D"):
        validate_inputs(v.reshape(10, 10), i, T)


def test_too_short():
    with pytest.raises(ValueError, match="at least 2"):
        validate_inputs([3.7], [0.0], [25.0])


def test_non_numeric():
    v, i, T = good()
    with pytest.raises(ValueError, match="real numbers"):
        validate_inputs(v.astype(str), i, T)


def test_short_gap_is_interpolated():
    v, i, T = good()
    v = v.copy()
    v[10:15] = np.nan            # 0.5 s gap at 10 Hz
    with pytest.warns(InputDomainWarning, match="interpolated 5"):
        v2, _, _ = validate_inputs(v, i, T)
    assert np.all(np.isfinite(v2)) and np.allclose(v2, 3.7)


def test_long_gap_rejected():
    v, i, T = good()
    i = i.copy()
    i[10:40] = np.nan            # 3 s gap
    with pytest.raises(ValueError, match="consecutive missing"):
        validate_inputs(v, i, T)


def test_all_missing_rejected():
    v, i, T = good()
    with pytest.raises(ValueError, match="no finite"):
        validate_inputs(v, i, np.full_like(T, np.nan))


@pytest.mark.parametrize("name, value, hint", [("v", 3700.0, "mV"), ("i", 2000.0, "mA"), ("T", 298.0, "kelvin")])
def test_wrong_units_rejected_with_hint(name, value, hint):
    v, i, T = good()
    arrs = dict(v=v, i=i, T=T)
    arrs[name] = np.full_like(v, value)
    with pytest.raises(ValueError, match=hint):
        validate_inputs(arrs["v"], arrs["i"], arrs["T"])


def test_out_of_validated_domain_warns():
    v, i, T = good()
    with pytest.warns(InputDomainWarning, match="validated range"):
        validate_inputs(v, i, np.full_like(T, -30.0))   # colder than any training data


def test_non_uniform_time_rejected():
    v, i, T = good()
    t = np.arange(100) * 0.1
    t[50:] += 0.05
    with pytest.raises(ValueError, match="non-uniform"):
        validate_inputs(v, i, T, t=t)


def test_non_increasing_time_rejected():
    v, i, T = good()
    t = np.arange(100) * 0.1
    t[50] = t[49]
    with pytest.raises(ValueError, match="increasing"):
        validate_inputs(v, i, T, t=t)
