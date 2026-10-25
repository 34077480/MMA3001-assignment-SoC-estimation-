# Report skeleton: write the prose yourself

Structure follows the brief's "Assessment Deliverables" list (8 items) and the rubric. Budget: 10 pages
max excluding references and appendices. Each section lists **what the marker looks for**, the **facts
and numbers** to use (all reproducible from `results/` via `notebooks/results.ipynb`), and the
**figure/table** to place. Bullets are notes, not text to paste. Write in your own words, and make sure
you can explain every claim without notes, since the interview is AI-free.

Suggested title: *State-of-charge estimation for EV batteries: an equivalent-circuit Extended Kalman
Filter versus a recurrent neural network*

---

## 1. Engineering problem (≈1 page)

Rubric: context, precise problem, why it matters, intended use, scope and limitations justified.

* BMS cannot measure SoC; infers it from V, I, T. Bad SoC: range anxiety, over-discharge, accelerated
  degradation, conservative usable-capacity margins. Automotive target often quoted as 2-3 % error.
* Intended use: an estimator that could run on a BMS microcontroller, per cell, in real time, with an
  uncertainty that downstream logic can use (e.g. conservative limits near empty).
* Scope: single cell (Panasonic NCR18650PF, 2.9 Ah, NCA chemistry), discharge-dominated drive cycles,
  ambient -20 to 25 °C, new-ish cell. **Out of scope**: pack-level balancing, charging (CC-CV),
  ageing / capacity fade (the cell lost 2.8 → 2.3 Ah over the whole campaign; we use early data),
  hysteresis modelling, real-vehicle sensors (emulated by corruption scenarios).

## 2. Inputs, outputs and data (≈1 page)

Rubric: inputs/outputs fully specified with format, units, domain, meaning, intended use; invalid input.

* Table: copy from README "Inputs and outputs" (V, I with sign convention, T; uniform dt; outputs soc,
  soc_raw, sigma).
* Invalid/missing/unsupported handling: summarise the `socest.validation` policy table (errors vs
  warnings, 1 s gap interpolation, unit hints).
* Data: Kollmeyer 18650PF (Mendeley, CC BY 4.0). 9 drive cycles × 5 temperatures, 10 Hz; C/20 and HPPC
  characterisation. Ground truth SoC = 1 + Ah/2.9 from the lab cycler; *note it is itself Coulomb
  counting with a precision shunt*: verified our integration reproduces the Ah counter to < 0.2 % SoC.
* Preprocessing: trimmed post-charge cool-down rests (15 files), resampled to 0.1 s, ~2 s logging gaps
  interpolated (all near 0 A).
* **Split** (table): train Cycles 1-4 + NN at 25/0/-20 °C; validation LA92 (all tuning and model
  selection); test US06/HWFET/UDDS at all 5 temperatures → 10 and -10 °C are unseen temperatures.

## 3. Computational approach (≈2 pages)

Rubric: how it works, why chosen, assumptions, settings, implementation, limitations.

* **OCV(SoC, T)**: C/20 discharge shape + level from relaxed HPPC rests; quadratic in T fitted on
  training temperatures only. Why not the standard charge/discharge average (65 mV bias). Fig: `ocv.png`.
* **ECM**: n-RC circuit; equations; parameters log-parameterised over SoC knots + Arrhenius in *cell*
  temperature (cell self-heats up to 17 °C in cold runs). Fitted by nonlinear least squares
  (scipy TRF, bounds, ftol convergence) with the lab SoC as input.
* **Discretisation**: exact zero-order hold; why (exact for piecewise-constant current, unconditionally
  stable, parameters valid at any sample rate) vs forward Euler (unstable for dt > 2τ).
* **EKF**: state [SoC, v1, v2]; Jacobians (give F[m,0] and H[0]); Joseph form; scalar innovation, so no
  matrix inverse. Tuning of q_s, q_v, r on validation with a **coverage constraint** (explain why
  RMSE-only tuning gave an overconfident filter). Sigma calibration factor.
* **LSTM/GRU**: inputs, 1 Hz, stateful; Gaussian head (mean, log-variance), beta-NLL; random start
  points (no "starts full" shortcut); augmentation with the same sensor model; TBPTT; early stopping
  on LA92. Needs no initial SoC.
* **Robustness scenarios** (table from `socest.sensors`).
* Assumptions/limitations list: no hysteresis state; parameters not identifiable uniquely
  (R0 vs fast RC split differs between fits with the same voltage error); fixed capacity; ML trained on
  15 sequences only.

## 4. Validation method and results (≈2 pages)

Rubric: appropriate validation design, correct metrics, comparison with ground truth / withheld data /
physical constraints, unsupported inputs and failure modes, honest limitations; say what validation
cannot establish and whether it influenced model selection.

* **Verification (code does what the maths says)**: ZOH vs analytical step response (1e-17 V),
  convergence orders 1.00 / 2.00, Euler instability; Jacobians vs finite differences (< 1e-7); numba
  EKF vs NumPy reference (< 1e-10); EKF on model-generated data converges from 30-point error.
  Fig: `integrator_verification.png`. Mention 52+ pytest tests, report in `docs/test_report.html`.
* **Validation (model matches reality)**: held-out cycles and temperatures; metrics RMSE, worst
  settled error (after 10 min), 3σ coverage, fraction outside [0, 1], charge bookkeeping.
  Acceptance: RMSE ≤ 3 %.
* Main results table (realistic sensors, test RMSE %). Selection was on validation: EKF order by
  validation RMSE (1-RC 2.40 %, 2-RC 1.45 %, **3-RC 1.07 %**), network by validation RMSE (LSTM-32 with a
  **100 s** BPTT window, 1.01 %):

  | | 25 | 10* | 0 | -10* | -20 |
  |---|---|---|---|---|---|
  | Coulomb counting | 8.83 | 8.92 | 9.35 | 9.47 | 9.58 |
  | EKF 1-RC | 3.06 | 2.88 | 3.13 | 3.41 | 4.46 |
  | EKF 2-RC | 1.24 | 1.10 | 1.89 | 2.62 | 2.89 |
  | EKF 3-RC (selected) | 1.30 | 1.65 | 1.48 | 2.21 | 3.40 |
  | LSTM-32 100 s, mean of 3 seeds | 1.23 | 1.60 | 1.70 | 2.86 | 2.42 |
  | (seed range) | 1.18-1.31 | 1.30-1.95 | 1.56-1.81 | 2.46-3.61 | 2.33-2.57 |
  | LSTM-32 100 s, 3-seed ensemble | 1.08 | 1.38 | 1.59 | 2.61 | 2.04 |
  | LSTM-32 500 s (first choice) | 1.16 | 1.48 | 1.77 | 3.02 | 3.49 |
  | GRU-32 500 s | 1.45 | 1.97 | 2.23 | 3.63 | 4.36 |

  Worst error after 10 min: EKF 3-RC 3.2 / 5.1 / 2.8 / 5.1 / 8.8 %; LSTM seed 0 4.9 / 4.3 / 4.8 / 7.7 / 8.1 %.

  (* unseen temperature). Figs: `accuracy_by_temperature.png`, `trajectories.png`, `robustness.png`.
* Points to make: CC is perfect on clean data (it *is* the ground truth) but drifts ~9 % with a cheap
  sensor; EKF recovers from 20-point initial error and ±50 mA bias; EKF and single LSTM are effectively
  tied on average and swap places by temperature; LSTM seed spread up to 1.15 points (-10 °C), as large
  as the method differences; selected EKF misses the 3 % target only at -20 °C, LSTM seeds at -10 °C.
* **Validation did not predict test ranking**: 3-RC was clearly best on validation (1.07 vs 1.45 %) but
  only tied with 2-RC on test and worse at -20 °C. Validation is LA92 at the *training* temperatures:
  it cannot see extrapolation to unseen/cold conditions. Lower voltage RMSE (3-RC 21 vs 27 mV at 25 °C)
  did not translate into better SoC. Discuss why (bias structure, not magnitude, drives SoC error).
* Failure mode: -20 °C US06, EKF 6-8 % low for the whole cycle and outside its own ±3σ
  (`trajectories.png`); the LSTM is noisy in the first minutes (no initial SoC) then recovers.
* Physical constraints: estimates outside [0, 1]: EKF 0.46 %, LSTM 1.20 %. EKF corrections are explicit
  (unexplained ΔSoC = total Kalman correction, 2.3 % on clean data); the LSTM has no charge-conservation
  structure (24 %, but dominated by its poor first estimate before it has seen data; say what the metric
  does and does not show).
* Uncertainty: 3σ coverage on test, raw / calibrated: EKF 78 / 75 %, LSTM 86 / 55 %. Both below the 95 %
  target; calibrating on mild validation conditions made the LSTM *worse* on cold/unseen test data.
* **What validation cannot establish**: aged cells, charging, other chemistries, real vehicle sensor
  behaviour (only emulated), temperatures below -20 °C, a pack; one cell only, so cell-to-cell
  variation is unknown. Validation (LA92) influenced every selection; test data influenced none.

## 5. Comparison and optimisation (≈2 pages)

Rubric: clear baseline, justified selection, quantitative accuracy/runtime/memory/robustness,
evidence-supported improvements, trade-offs.

* Baselines: Coulomb counting (and the first versions of each model).
* **Before → after** (fig `before_after.png`): OCV fix (65 → 4.5 mV); residual sampling 1 → 5 Hz
  (cold RMSE -0.5 points, more physical R0); RMSE-only → coverage-constrained tuning (coverage ~45 % →
  95 % on validation for +0.07 points RMSE); LSTM training changes: **stabilised training but no
  significant test improvement** (say so).
* Sensitivity (fig `sensitivity.png`): covariance surface (RMSE flat over a wide band of r; coverage is
  what separates settings); RC pairs test RMSE 1/2/3-RC = 3.39 / 1.95 / 2.01 %; sampling rate (selected
  EKF, same parameters thanks to exact ZOH) 10 / 5 / 2 / 1 / 0.5 Hz = 2.01 / 1.88 / 1.91 / 2.16 / 2.63 %,
  so 2 Hz is 5× cheaper at no accuracy cost; parameter tornado: capacity -5 % (+0.79), R1 ×1.2 (+0.62),
  R0 ×1.2 (+0.54); time constants negligible; capacity matters most (link to ageing: 2.8 → 2.3 Ah);
  LSTM window 100 / 250 / 500 / 1000 s = 1.81 / 1.87 / 2.18 / 3.04 % (confounded: shorter window = more
  optimiser updates per epoch); hidden 16 / 32 / 64 = 2.65 / 2.18 / 2.25 %.
* Cost (table from `perf.json`): EKF 1/2/3-RC 218/460/860 FLOP per step at 10 Hz (2.2/4.6/8.6 kFLOP/s);
  LSTM-16/32/64 2.7k/9.5k/35k FLOP per step at 1 Hz; arithmetic intensity LSTM ~0.5 FLOP/byte (memory
  bound: every weight read each step), EKF 0.7-1.2; MCU estimate (Cortex-M4F 168 MHz) EKF 3-RC 9 µs,
  LSTM-32 114 µs per update, <0.02 % CPU for one cell (×~100 for a pack). Numba EKF 0.20 µs/step vs
  NumPy reference 39 µs (196×); PyTorch per-step calls 172 µs vs 2.6 µs as a compiled loop (framework
  overhead dominates). Profiler-driven fix: cProfile showed 93 % of end-to-end `estimate_soc` time
  (1.05 of 1.12 s) was rebuilding OCV from raw files; caching the fitted OCV model cut it to 0.08 s
  (0.07 s is the EKF). Compare `results/profile_ekf_pipeline.txt` before/after in the git history.
* Pareto figure `pareto.png`: accuracy vs compute, physics vs black box.
* **Final design justification**: your call, argued from the evidence. Facts: average accuracy tied;
  EKF has lower worst-case error at 3 of 5 temperatures (within 1 point at the other two), interpretable states, explicit charge
  bookkeeping, is robust to initial-SoC and bias errors, and runs at 2 Hz for ~1.7 kFLOP/s; the LSTM
  needs no initial SoC and no characterisation tests, but has large seed variance, poor uncertainty
  calibration and no physical structure; the ensemble is most accurate at 3× the cost.

## 6. Limitations and lessons learned (≈1 page)

* Model-form error dominates EKF error (cycle-dependent voltage bias, US06 high current); no
  hysteresis; parameter non-identifiability; temperature extrapolation.
* Uncertainty calibration does not transfer to harder conditions.
* ML data-hungry: 15 training sequences; high seed variance.
* Lessons (your own): e.g. "a textbook shortcut (OCV averaging) was wrong for this cell"; "a
  file-matching bug silently leaked test data: validation must check data provenance"; "RMSE-only
  tuning hides overconfidence"; "a memory workaround changed results".
* Improvements, well justified: hysteresis state; current-dependent R0; parameter fitting on HPPC +
  drive cycles to fix identifiability; adaptive/dual EKF for ageing; physics-informed features or
  hybrid (EKF + learned residual); conformal calibration of uncertainty; smaller OCV table for MCU.

## 7. AI-use reflection (≈1 page)

Use `docs/report/ai_use_log.md` for the facts. Cover every bullet in brief section 7: tools, what for,
level of contribution (be honest: most code was AI-written), why, how checked, which decisions were
yours, AI errors (the log lists 11), effect on understanding/workflow.

## 8. References

* Kollmeyer, P. (2018) Panasonic 18650PF Li-ion Battery Data, Mendeley Data, V1, doi:10.17632/wykht8y7tg.1.
* Plett, G. L. (2004) Extended Kalman filtering for battery management systems of LiPB-based HEV
  battery packs, Parts 1-3, J. Power Sources 134(2).
* Chemali, E., Kollmeyer, P. J., Preindl, M., Ahmed, R., Emadi, A. (2018) Long short-term memory
  networks for accurate state-of-charge estimation of Li-ion batteries, IEEE Trans. Ind. Electron.
  65(8).
* Seitzer, M., Tavakoli, A., Antic, D., Martius, G. (2022) On the pitfalls of heteroscedastic
  uncertainty estimation with probabilistic neural networks, ICLR.
* Hochreiter, S., Schmidhuber, J. (1997) Long short-term memory, Neural Computation 9(8).
* Virtanen, P. et al. (2020) SciPy 1.0, Nature Methods 17; Lam, S. K. et al. (2015) Numba, LLVM-HPC.

*(Check each reference yourself before citing it; do not cite anything you have not looked at.)*

## Appendices (not counted)

* A: full results tables per cycle and scenario (from the notebook).
* B: test report summary; repository structure.
