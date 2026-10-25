"""Loading the Panasonic 18650PF dataset (Kollmeyer, Mendeley Data, doi:10.17632/wykht8y7tg.1).

Sign convention (from the dataset): current < 0 is discharge. The Ah counter is reset at the
start of each drive cycle, which begins from a full charge, so ground-truth SoC is

    SoC(t) = 1 + Ah(t) / Q_REF

with Q_REF = 2.9 Ah, the reference capacity the dataset readme uses to define its DOD cutoffs.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Panasonic 18650PF Data"
Q_REF = 2.9  # Ah
DT = 0.1     # s, nominal drive-cycle sample period

TEMP_DIRS = {
    25: RAW / "Panasonic 18650PF Data" / "25degC",
    10: RAW / "10degC",
    0: RAW / "0degC",
    -10: RAW / "-10degC",
    -20: RAW / "-20degC",
}
CYCLES = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "US06", "HWFET", "UDDS", "LA92", "NN"]

# Train/validation/test split. Models are fitted only on TRAIN cycles at TRAIN_TEMPS; LA92 is used for
# tuning (EKF covariances, ML early stopping); the EPA cycles are held out and are evaluated at every
# temperature, so 10C and -10C test both unseen cycles and unseen temperatures.
TRAIN_TEMPS = [25, 0, -20]
UNSEEN_TEMPS = [10, -10]
TRAIN_CYCLES = ["Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN"]
VAL_CYCLES = ["LA92"]
TEST_CYCLES = ["US06", "HWFET", "UDDS"]


@dataclass
class Run:
    name: str
    temp: int          # ambient temperature, degC
    t: np.ndarray      # s
    v: np.ndarray      # V
    i: np.ndarray      # A, discharge negative
    T: np.ndarray      # battery case temperature, degC
    ah: np.ndarray     # Ah, from the lab cycler

    @property
    def soc(self):
        return 1.0 + self.ah / Q_REF


def load_mat(path):
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["meas"]
    return dict(t=np.asarray(m.Time, float), v=np.asarray(m.Voltage, float),
                i=np.asarray(m.Current, float), T=np.asarray(m.Battery_Temp_degC, float),
                ah=np.asarray(m.Ah, float))


def find_cycle_file(temp, cycle):
    """Return the split single-cycle file for (temp, cycle); the HWFET name varies (HWFTa/HWFTb at 25C)."""
    folder = next(p for p in TEMP_DIRS[temp].iterdir() if p.is_dir() and p.name.lower().startswith("drive"))
    prefix = f"{'n' if temp < 0 else ''}{abs(temp)}degC_"
    pat = {"HWFET": r"HWF(ET|Ta)"}.get(cycle, re.escape(cycle))
    hits = [p for p in folder.glob("*.mat")
            if re.search(rf" {re.escape(prefix)}{pat}_Pan18650PF\.mat$", p.name)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{temp}C {cycle}: {[h.name for h in hits]}")
    return hits[0]


def load_cycle(temp, cycle, dt=DT):
    """Load one drive cycle, resampled onto a uniform grid of period dt."""
    d = load_mat(find_cycle_file(temp, cycle))
    # Some split files open with the post-charge cool-down rest (0 A, logged once a minute);
    # start at the first fast-logged sample so every run begins at the drive cycle itself.
    start = int(np.argmax(np.diff(d["t"]) < 0.5))
    d = {k: x[start:] for k, x in d.items()}
    t = d["t"]
    keep = np.concatenate([[True], np.diff(t) > 0])  # drop duplicate timestamps
    tg = np.arange(t[0], t[keep][-1], dt)
    res = {k: np.interp(tg, t[keep], d[k][keep]) for k in ("v", "i", "T", "ah")}
    return Run(f"{temp}C_{cycle}", temp, tg - tg[0], **res)


def all_cycles(temps=TEMP_DIRS, cycles=CYCLES):
    return {(T, c): load_cycle(T, c) for T in temps for c in cycles}
