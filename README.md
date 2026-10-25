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

Test RMSE [% SoC] on the held-out EPA cycles with a simulated low-cost sensor (10-point initial SoC
error, 30 mA offset, 1 % gain error, noise). Model choices were made on the validation cycle only:
validation selected the **3-RC EKF** and an **LSTM-32 trained with a 100 s BPTT window**.

| | 25 °C | 10 °C (unseen) | 0 °C | -10 °C (unseen) | -20 °C |
|---|---|---|---|---|---|
| Coulomb counting | 8.83 | 8.92 | 9.35 | 9.47 | 9.58 |
| EKF 2-RC | 1.24 | 1.10 | 1.89 | 2.62 | 2.89 |
| EKF 3-RC (selected) | 1.30 | 1.65 | 1.48 | 2.21 | 3.40 |
| LSTM-32, mean of 3 seeds (range) | 1.23 (1.18-1.31) | 1.60 (1.30-1.95) | 1.70 (1.56-1.81) | 2.86 (2.46-3.61) | 2.42 (2.33-2.57) |
| LSTM-32, 3-seed ensemble | 1.08 | 1.38 | 1.59 | 2.61 | 2.04 |

* Coulomb counting is exact on clean lab data (the ground truth is Coulomb counting on a precision
  shunt) but drifts to ~9 % with a realistic sensor; both the EKF and the LSTM stay near 1-3 %.
* The EKF and a single LSTM are effectively tied on average (no significance test was run); which wins depends on temperature, and
  LSTM seed-to-seed spread (up to 1.15 points) is as large as the differences between methods.
* Worst-case error after convergence: EKF 2.8-8.8 %, LSTM 4.3-8.1 %. The EKF fails on -20 °C US06
  (6-8 % low for the whole cycle, outside its own uncertainty band).
* Cost per second of driving: EKF 3-RC 8.6 kFLOP at 10 Hz (and no accuracy loss at 2 Hz, i.e. 1.7 kFLOP),
  LSTM-32 9.5 kFLOP at 1 Hz. The numba EKF (2-RC) is ~196× faster than the NumPy reference with identical output.
* Most influential EKF inputs: assumed capacity (-5 % → +0.8 points RMSE), then R1 and R0.
* Neither method's 3σ uncertainty reaches the 95 % coverage target on the test set.

Figures are in `results/figures/`; every number above is recomputed in `notebooks/results.ipynb`.

## AI use

This project was developed with guidance and assistance from Claude (Anthropic) via Claude Code, primarily for problem-solving, debugging, and code improvement. The implementation was carried out by the author, who reviewed, adapted, and tested the suggested outputs. The nature and extent of AI assistance, the verification process, and any errors encountered are described in the report’s AI-use reflection.

## Licence and citation

Code: MIT (see `LICENSE`). Data: P. Kollmeyer, "Panasonic 18650PF Li-ion Battery Data", Mendeley Data,
V1, 2018, doi:10.17632/wykht8y7tg.1 (CC BY 4.0).
