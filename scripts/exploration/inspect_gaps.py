"""Check slow-logged (>1 s) steps left inside each drive cycle after trimming the leading rest."""
import numpy as np

from socest.data import CYCLES, TEMP_DIRS, find_cycle_file, load_mat

for T in TEMP_DIRS:
    for c in CYCLES:
        d = load_mat(find_cycle_file(T, c))
        start = int(np.argmax(np.diff(d["t"]) < 0.5))
        t, i = d["t"][start:], d["i"][start:]
        dt = np.diff(t)
        slow = np.where(dt > 1.0)[0]
        if len(slow):
            # current on both sides of each slow step: if ~0 the cell was resting and interpolation is harmless
            imax = max(np.abs(i[slow]).max(), np.abs(i[slow + 1]).max())
            print(f"{T:>4}C {c:8s} {len(slow):4d} slow steps, longest {dt.max():6.1f}s, "
                  f"total {dt[slow].sum()/60:6.1f} min, at t={t[slow[0]]:.0f}..{t[slow[-1]+1]:.0f}s "
                  f"of {t[-1]-t[0]:.0f}s, |I| at gaps <= {imax:.3f} A")
