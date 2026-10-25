# Li-ion state-of-charge estimation: physics-based EKF vs recurrent neural network

MMA3001 Numerical Methods and Machine Learning, individual project.

An electric vehicle's battery management system (BMS) cannot measure state of charge (SoC) directly; it
must infer it from terminal voltage, current and temperature. This repository builds, verifies and
compares three estimators on real automotive drive-cycle data:

| Method | Type | Needs initial SoC? | Interpretable states |
|---|---|---|---|
| Coulomb counting | integrate current (baseline) | yes | yes |
| Extended Kalman Filter on a 1/2/3-RC equivalent circuit model | numerical: ODE discretisation, nonlinear least squares, linearisation | yes (corrects it) | yes (SoC, RC voltages) |
| LSTM / GRU with a Gaussian uncertainty head | machine learning | no | no |

**Headline results** are summarised in [Results](#results); the full analysis, figures and limitations are in
the project report (submitted separately) and in `notebooks/results.ipynb`.

## Inputs and outputs

**Top-level function:** `socest.api.estimate_soc(v, i, T, dt=0.1, method="ekf", soc0=None)`

| Input | Units | Format / domain |
|---|---|---|
| `v` terminal voltage | V | 1-D float array; physical limits 0-5 V, validated range 2.4-4.25 V |
| `i` current | A, **discharge negative** | same length; limits ±100 A, validated -21.5 to +11 A |
| `T` cell (case) temperature | °C | same length; limits -60 to 100 °C, validated -21 to 34 °C |
| `dt` sample period | s | uniform grid; EKF any `dt`, ML model requires 0.1 s |
| `soc0` (optional) | fraction | initial SoC for EKF / Coulomb counting; if omitted, taken from the OCV of the first sample (assumes a rested cell) |

| Output | Meaning and intended use |
|---|---|
| `soc` | SoC estimate, fraction, clipped to the physical range [0, 1]; for range display and charge/discharge limits |
| `soc_raw` | unclipped estimate (for diagnostics: how often an estimator leaves [0, 1]) |
| `sigma` | calibrated 1-σ uncertainty (fraction); for conservative limits, e.g. use `soc - 3*sigma` near empty |

**Invalid, missing or unsupported input** (`socest.validation`): mismatched lengths, non-numeric data,
non-uniform time stamps and values outside physical limits raise `ValueError` (the message suggests likely
unit mistakes such as mV or kelvin). Gaps of missing samples up to 1 s are linearly interpolated with an
`InputDomainWarning`; longer gaps raise. Values inside the physical limits but outside the validated
range give an `InputDomainWarning`: the estimate is then an extrapolation.

## Repository layout

```
src/socest/          library (installable package)
  data.py            dataset loading, resampling, train/validation/test split
  ocv.py             open-circuit voltage OCV(SoC, T) from C/20 and HPPC tests
  ecm.py             n-RC equivalent circuit model, exact ZOH discretisation (numba)
  ekf.py             Coulomb counting, EKF (numba) and a readable NumPy reference EKF
  ml.py              LSTM/GRU estimator, training loop, beta-NLL loss
  sensors.py         sensor-corruption scenarios for robustness testing
  metrics.py         error metrics and physical sanity checks
  validation.py      input checking and cleaning
  api.py             estimate_soc(): the top-level entry point
scripts/             pipeline steps (run_all.py runs everything in order)
  exploration/       development and debugging scripts, kept as a record (see its README)
tests/               pytest suite (verification of the numerics and of input handling)
notebooks/           results walkthrough (executed)
results/             fitted parameters, trained networks, evaluation JSON, figures, profiles
docs/api/            HTML API documentation (pdoc)
docs/test_report.html  pytest report
data/raw/            dataset goes here (not committed)
```

## Setup

Tested with Python 3.10.5 on Windows 11, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

### Data

The dataset is **not** included (198 MB, separate licence). Download
*Panasonic 18650PF Li-ion Battery Data* (P. Kollmeyer, Mendeley Data, doi:10.17632/wykht8y7tg.1,
CC BY 4.0) with "Download All" from https://data.mendeley.com/datasets/wykht8y7tg/1, then:

```bash
python scripts/get_data.py path/to/wykht8y7tg-1.zip
```

## Reproducing the results

```bash
python scripts/run_all.py                          # everything from raw data (~1.5-2 h, CPU)
python scripts/run_all.py --skip-fit --skip-train  # reuse committed parameters and networks (~10 min)
python -m pytest                                   # tests only (~15 s; data tests skip without the dataset)
```

Individual steps: `fit_ecm.py` → `run_ekf.py` → `train_ml.py` → `run_ml.py` → `sensitivity.py` →
`perf.py` → `make_figures.py`. Every script documents its inputs and outputs in its docstring.

**Data split.** Parameters are fitted on Cycles 1-4 and NN at 25, 0 and -20 °C. LA92 (at the same
temperatures) is used for every tuning and model-selection decision. The EPA cycles US06, HWFET and UDDS
are held out and evaluated at all five temperatures, so 10 °C and -10 °C test unseen cycles **and** unseen
temperatures.

## Verification and validation (summary)

* Integrator: exact ZOH matches the analytical RC step response to 1e-17 V; forward Euler shows order 1.00
  and instability beyond dt = 2τ; Tustin order 2.00 (`tests/test_integrator.py`).
* EKF Jacobians match central finite differences to < 1e-7; the numba EKF equals an independent NumPy
  implementation to < 1e-10; on model-generated data the EKF converges from a 30-point initial error.
* OCV model vs relaxed HPPC rests: 4.5 mV RMSE at 25 °C, 7-8 mV at unseen temperatures.
* Ground truth: our integration of the logged current reproduces the cycler's Ah counter to < 0.2 % SoC.
* Robustness: wrong initial SoC, ±50 mA current bias, sensor noise and a combined "realistic" sensor,
  at all five temperatures.

## Results

*(filled in from `results/` after the final run; see `notebooks/results.ipynb`)*

## AI use

This project was developed with substantial assistance from Claude (Anthropic) via Claude Code. Commits
co-written with the assistant carry a `Co-Authored-By` trailer. The nature and extent of AI use, how its
output was checked, and the errors it made are described in the report's AI-use reflection.

## Licence and citation

Code: MIT (see `LICENSE`). Data: P. Kollmeyer, "Panasonic 18650PF Li-ion Battery Data", Mendeley Data,
V1, 2018, doi:10.17632/wykht8y7tg.1 (CC BY 4.0).
