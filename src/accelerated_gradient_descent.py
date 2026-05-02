import numpy as np
from . import potential_field as pf
from .line_search import backtracking_line_search


def plan(env, t0=1.0, alpha=1e-4, beta=0.5, max_iter=2000, tol=0.05, d0=2.0, restart=True):
    p = env.start.copy()
    p_prev = p.copy()
    path = [p.copy()]

    def f(x):
        return pf.total(x, env.goal, env.obstacles, d0=d0)
    
    def grad(x):
        return pf.total_grad(x, env.goal, env.obstacles, d0=d0)
    
    for k in range(max_iter):
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, k
        momentum = k / (k + 3)
        y = p + momentum * (p - p_prev)
        g = grad(y)
        if np.linalg.norm(g) < 1e-10:
            return np.array(path), False, k
        direction = -g
        step = backtracking_line_search(f, grad, y, direction, t0=t0, alpha=alpha, beta=beta)
        p_next = y + step * direction
        
        #fall back to plain GD if acceleration makwes it worse
        if restart and f(p_next) > f(p):
            y = p.copy()
            g = grad(y)
            direction = -g
            step = backtracking_line_search(f, grad, y, direction, t0=t0, alpha=alpha, beta=beta)
            p_next = y + step * direction
            p_prev = p.copy()
        else:
            p_prev = p.copy()
        p = p_next
        path.append(p.copy())

    return np.array(path), False, max_iter