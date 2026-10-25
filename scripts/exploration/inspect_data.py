"""Sanity summary of every drive-cycle file: duration, sample rate, ranges, and a Coulomb-count check."""
import numpy as np

from socest.data import CYCLES, TEMP_DIRS, find_cycle_file, load_cycle, load_mat

hdr = ("run", "dur_min", "dt_med", "dt_max", "V0", "Vmin", "Ah0", "Ah_end", "SoC_end",
       "Imin", "Imax", "Tmin", "Tmax", "CC_err_mAh")
print(("{:14s}" + " {:>8s}" * (len(hdr) - 1)).format(*hdr))
for T in TEMP_DIRS:
    for c in CYCLES:
        try:
            raw = load_mat(find_cycle_file(T, c))
        except FileNotFoundError as e:
            print(f"{T}C {c}: MISSING {e}")
            continue
        r = load_cycle(T, c)
        dtr = np.diff(raw["t"])
        # integrate current ourselves and compare with the cycler's Ah counter
        cc = np.concatenate([[0], np.cumsum(r.i[:-1] * np.diff(r.t))]) / 3600
        err = np.abs(cc - (r.ah - r.ah[0])).max() * 1000
        vals = (r.t[-1] / 60, np.median(dtr), dtr.max(), r.v[0], r.v.min(), raw["ah"][0], r.ah[-1],
                r.soc[-1], r.i.min(), r.i.max(), r.T.min(), r.T.max(), err)
        print(f"{r.name:14s}" + "".join(f" {x:8.3f}" for x in vals))
