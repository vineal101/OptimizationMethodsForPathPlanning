import numpy as np
from . import potential_field as pf


def plan(env, step=0.05, max_iter=2000, tol=0.05):
    p = env.start.copy()
    path = [p.copy()]
    for i in range(max_iter):
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
        g = pf.total_grad(p, env.goal, env.obstacles)
        p = p - step * g
        path.append(p.copy())
    return np.array(path), False, max_iter
