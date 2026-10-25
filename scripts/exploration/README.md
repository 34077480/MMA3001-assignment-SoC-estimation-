# Exploration and debugging scripts

Kept as a record of how the data was understood and how bugs were found during development.
They are not part of the pipeline. Run from the repository root, e.g. `python scripts/exploration/inspect_data.py`.

| Script | What it showed |
|---|---|
| `inspect_data.py` | All 45 drive-cycle files load; our integration of current matches the cycler's Ah counter to < 5 mAh; SoC end points match the readme's DOD cut-offs. |
| `inspect_gaps.py` | 15 files start with a 1-min-logged post-charge rest (cell cooling from ~17 degC); remaining gaps are ~2 s at near-zero current. Led to trimming in `load_cycle`. |
| `inspect_char.py` | Structure of the C/20 and HPPC characterisation files. |
| `check_ocv.py` | OCV model vs relaxed HPPC rests; first version (C/20 charge/discharge average) was 64 mV off. |
| `debug_ocv.py` | Found that 10, 0 and -10 degC returned identical data: the file matcher let `0degC` match `n10degC` (unseen-temperature leak). |
| `debug_ecm.py` | 1-RC residuals: slow polarisation at 0 degC the model could not represent; motivated 2-RC and denser low-SoC knots. |
| `debug_bias.py` | Voltage bias by SoC/temperature/cycle: cycle-dependent (US06), i.e. current nonlinearity and long memory. |
| `debug_jac.py` | Located a Jacobian mismatch in dOCV/ds (smoothed slope vs interpolant slope). *Historical: written against the prototype test module (first commit) and no longer runs; the check now lives in `tests/test_ekf.py`.* |
