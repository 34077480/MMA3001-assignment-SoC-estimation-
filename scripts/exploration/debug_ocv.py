import numpy as np

from socest.ocv import OCV, hppc_rest_points

ocv = OCV()
p25 = hppc_rest_points(25)
for T in [25, 10, 0, -10, -20]:
    p = hppc_rest_points(T)
    ref = np.interp(p[:, 0], p25[::-1, 0], p25[::-1, 1])
    print(f"{T:>4}C  SoC: " + " ".join(f"{s:5.2f}" for s in p[:, 0]))
    print(f"      V-V25 mV: " + " ".join(f"{x:5.0f}" for x in (p[:, 1] - ref) * 1000))
    print(f"      model mV: " + " ".join(f"{x:5.0f}" for x in (ocv(p[:, 0], T) - ocv(p[:, 0], 25)) * 1000))
