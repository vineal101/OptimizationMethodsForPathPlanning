import numpy as np
from . import potential_field as pf
from .line_search import backtracking_line_search


def plan(env, t0=1.0, alpha=1e-4, beta=0.5, max_iter=2000, tol=0.05, d0=2.0):
    p = env.start.copy()
    path = [p.copy()]
    
    def f(x):
        return pf.total(x, env.goal, env.obstacles, d0=d0)
    
    def grad(x):
        return pf.total_grad(x, env.goal, env.obstacles, d0=d0)
    
    for i in range(max_iter):
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
        g = grad(p)
        if np.linalg.norm(g) < 1e-10:
            return np.array(path), False, i
        direction = -g
        step = backtracking_line_search(f, grad, p, direction, t0=t0, alpha=alpha, beta=beta)
        p = p + step * direction
        path.append(p.copy())
    return np.array(path), False, max_iter