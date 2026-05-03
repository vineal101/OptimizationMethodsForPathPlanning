import numpy as np
from src import trajectory_costs as tc

np.random.seed(0)
path = np.random.randn(10, 2)

g_analytical = tc.smoothness_grad(path)
g_numerical = np.zeros_like(path)
eps = 1e-6
for i in range(path.shape[0]):
    for j in range(path.shape[1]):
        path[i, j] += eps
        f_plus = tc.smoothness(path)
        path[i, j] -= 2 * eps
        f_minus = tc.smoothness(path)
        path[i, j] += eps
        g_numerical[i, j] = (f_plus - f_minus) / (2 * eps)

err = np.max(np.abs(g_analytical - g_numerical))
print(f"Smoothness gradient max error: {err:.2e}")
