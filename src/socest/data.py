"""Loading and preprocessing of the Panasonic 18650PF dataset.

Dataset: P. Kollmeyer, "Panasonic 18650PF Li-ion Battery Data", Mendeley Data, V1, 2018,
doi:10.17632/wykht8y7tg.1, licence CC BY 4.0. See the README for download instructions.

Conventions
-----------
* Current is in amperes with **discharge negative** (the dataset's convention).
* The cycler's amp-hour counter is reset at the start of every drive cycle, and every drive cycle starts
  from a full charge, so the ground-truth state of charge is

      SoC(t) = 1 + Ah(t) / Q_REF,      Q_REF = 2.9 Ah,

  the reference capacity the dataset readme uses to define its depth-of-discharge cut-offs
  (e.g. 2.32 Ah = 80 % DOD at 0 degC).
* All drive cycles are resampled onto a uniform 0.1 s grid (``DT``).

Train / validation / test split
-------------------------------
Model parameters are fitted only on ``TRAIN_CYCLES`` at ``TRAIN_TEMPS``. ``VAL_CYCLES`` (LA92) is used for
tuning (EKF covariances, ML early stopping, uncertainty calibration). ``TEST_CYCLES`` (the standard EPA
cycles) are never used for any decision and are evaluated at all five temperatures, so 10 degC and
-10 degC test unseen cycles *and* unseen temperatures.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
"""Repository root."""
RAW = ROOT / "data" / "raw" / "Panasonic 18650PF Data"
"""Folder created by extracting the Mendeley zip into ``data/raw``."""
RESULTS = ROOT / "results"
"""Folder for fitted parameters, evaluation JSON files and figures."""

Q_REF = 2.9
"""Reference capacity [Ah] used to define SoC."""
DT = 0.1
"""Sample period [s] of the resampled drive cycles (10 Hz)."""

TEMP_DIRS = {
    25: RAW / "Panasonic 18650PF Data" / "25degC",
    10: RAW / "10degC",
    0: RAW / "0degC",
    -10: RAW / "-10degC",
    -20: RAW / "-20degC",
}
"""Ambient temperature [degC] -> data folder."""
CYCLES = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "US06", "HWFET", "UDDS", "LA92", "NN"]
"""The nine drive cycles recorded at every temperature."""

TRAIN_TEMPS = [25, 0, -20]
UNSEEN_TEMPS = [10, -10]
TRAIN_CYCLES = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN"]
VAL_CYCLES = ["LA92"]
TEST_CYCLES = ["US06", "HWFET", "UDDS"]


@dataclass
class Run:
    """One drive cycle on a uniform time grid.

    Attributes
    ----------
    name : str
        Identifier such as ``"25C_UDDS"``.
    temp : int
        Ambient (chamber) temperature [degC].
    t : numpy.ndarray
        Time since the start of the cycle [s].
    v : numpy.ndarray
        Terminal voltage [V].
    i : numpy.ndarray
        Current [A], discharge negative.
    T : numpy.ndarray
        Cell case temperature [degC] (thermocouple at mid-height of the cell).
    ah : numpy.ndarray
        Amp-hours from the cycler since the start of the cycle [Ah].
    """
    name: str
    temp: int
    t: np.ndarray
    v: np.ndarray
    i: np.ndarray
    T: np.ndarray
    ah: np.ndarray

    @property
    def soc(self):
        """numpy.ndarray: ground-truth SoC (fraction, nominally 0-1) from the lab amp-hour counter."""
        return 1.0 + self.ah / Q_REF


def _check_raw():
    if not RAW.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW}. Download the Mendeley zip and run "
            "`python scripts/get_data.py <path-to-zip>` (see README).")


def load_mat(path):
    """Read one dataset ``.mat`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a file from the dataset (each contains a ``meas`` struct).

    Returns
    -------
    dict of numpy.ndarray
        Keys ``t`` [s], ``v`` [V], ``i`` [A], ``T`` [degC] and ``ah`` [Ah], as logged (not resampled).

    Raises
    ------
    KeyError
        If the file has no ``meas`` struct (not a file from this dataset).
    """
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["meas"]
    return dict(t=np.asarray(m.Time, float), v=np.asarray(m.Voltage, float),
                i=np.asarray(m.Current, float), T=np.asarray(m.Battery_Temp_degC, float),
                ah=np.asarray(m.Ah, float))


def find_cycle_file(temp, cycle):
    """Locate the split single-cycle file for one temperature and drive cycle.

    File names look like ``"03-21-17_00.29 25degC_UDDS_Pan18650PF.mat"``. The match is anchored on the
    space before the descriptor so that ``0degC`` cannot match ``n10degC`` (a bug in an earlier version).
    The 25 degC HWFET file is named ``HWFTa``.

    Parameters
    ----------
    temp : int
        Ambient temperature, one of ``TEMP_DIRS``.
    cycle : str
        One of ``CYCLES``.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    KeyError
        If ``temp`` is not a recorded temperature.
    FileNotFoundError
        If no file, or more than one file, matches.
    """
    _check_raw()
    if temp not in TEMP_DIRS:
        raise KeyError(f"temperature {temp} not in dataset; choose from {list(TEMP_DIRS)}")
    folder = next(p for p in TEMP_DIRS[temp].iterdir() if p.is_dir() and p.name.lower().startswith("drive"))
    prefix = f"{'n' if temp < 0 else ''}{abs(temp)}degC_"
    pat = {"HWFET": r"HWF(ET|Ta)"}.get(cycle, re.escape(cycle))
    hits = [p for p in folder.glob("*.mat")
            if re.search(rf" {re.escape(prefix)}{pat}_Pan18650PF\.mat$", p.name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{temp}C {cycle}: expected one file, found {[h.name for h in hits]}")
    return hits[0]


def load_cycle(temp, cycle, dt=DT):
    """Load one drive cycle, trimmed to the drive cycle itself and resampled to a uniform grid.

    Some split files open with the post-charge cool-down rest (0 A, logged once a minute); the run starts
    at the first fast-logged (0.1 s) sample. Duplicate timestamps are dropped and the remaining ~2 s
    logging gaps (all at near-zero current) are bridged by linear interpolation.

    Parameters
    ----------
    temp : int
        Ambient temperature [degC], one of ``TEMP_DIRS``.
    cycle : str
        Drive cycle name, one of ``CYCLES``.
    dt : float, optional
        Output sample period [s]. Default ``DT`` (0.1 s).

    Returns
    -------
    Run
    """
    d = load_mat(find_cycle_file(temp, cycle))
    start = int(np.argmax(np.diff(d["t"]) < 0.5))
    d = {k: x[start:] for k, x in d.items()}
    t = d["t"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    tg = np.arange(t[0], t[keep][-1], dt)
    res = {k: np.interp(tg, t[keep], d[k][keep]) for k in ("v", "i", "T", "ah")}
    return Run(f"{temp}C_{cycle}", temp, tg - tg[0], **res)
