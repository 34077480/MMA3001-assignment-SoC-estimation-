"""socest: state-of-charge estimation for Li-ion cells, physics-based EKF vs recurrent neural network.

Modules
-------
- :mod:`socest.data`        dataset loading, resampling, train/validation/test split
- :mod:`socest.ocv`         open-circuit voltage OCV(SoC, T) from the characterisation tests
- :mod:`socest.ecm`         n-RC equivalent circuit model (exact ZOH discretisation, numba kernels)
- :mod:`socest.ekf`         Coulomb counting baseline and Extended Kalman Filter
- :mod:`socest.ml`          LSTM/GRU estimator with a Gaussian uncertainty head
- :mod:`socest.sensors`     sensor-corruption scenarios for robustness tests
- :mod:`socest.metrics`     error metrics and physical sanity checks
- :mod:`socest.validation`  checking and cleaning of inputs
- :mod:`socest.api`         ``estimate_soc(v, i, T)``: the top-level entry point
"""
__version__ = "1.0.0"
