"""HPPC files: rest voltage before each pulse group (approx. OCV) vs SoC, per temperature."""
import numpy as np

from socest.data import Q_REF, RAW, load_mat

for f in sorted(RAW.rglob("*5*ulse*HPPC*.mat")):
    d = load_mat(f)
    t, i, v, ah = d["t"], d["i"], d["v"], d["ah"]
    active = np.abs(i) > 0.02
    starts = np.flatnonzero(np.diff(active.astype(int)) == 1) + 1
    out = []
    prev_end = 0
    for s in starts:
        rest = t[s - 1] - t[prev_end] if prev_end else np.nan
        if rest > 300 or prev_end == 0:   # a long rest -> start of a new pulse group
            out.append((1 + ah[s - 1] / Q_REF, v[s - 1], rest / 60))
        ends = np.flatnonzero(~active[s:])
        prev_end = s + (ends[0] if len(ends) else 0)
    print(f.name.split(" ")[1], " ".join(f"{soc:.2f}:{ocv:.3f}({r:.0f}m)" for soc, ocv, r in out))
