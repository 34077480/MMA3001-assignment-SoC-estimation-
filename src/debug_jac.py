import sys
sys.path.insert(0, "tests")
import numpy as np
from test_ekf import analytic, numeric, ROOT

theta = np.load(ROOT / "results" / "ecm_theta_2rc.npy")
rng = np.random.default_rng(1)
worst = {}
for _ in range(300):
    s = rng.uniform(0.05, 0.95)
    x = np.array([s, rng.uniform(-0.1, 0), rng.uniform(-0.1, 0)])
    i, T = rng.uniform(-15, 8), rng.uniform(-20, 30)
    Ha, Fa = analytic(theta, x, i, T)
    Hn, Fn = numeric(theta, x, i, T)
    for name, a, n in [("H0", Ha[0], Hn[0]), ("F10", Fa[1, 0], Fn[1, 0]), ("F20", Fa[2, 0], Fn[2, 0]),
                       ("F11", Fa[1, 1], Fn[1, 1]), ("F22", Fa[2, 2], Fn[2, 2])]:
        e = abs(a - n) / max(abs(n), 1e-3)
        if e > worst.get(name, (0,))[0]:
            worst[name] = (e, a, n, s, i, T)
for k, v in worst.items():
    print(k, "rel err %.2e  analytic %.5g  numeric %.5g  at s=%.4f i=%.1f T=%.1f" % v)
