import numpy as np
from . import potential_field as pf
from . import sdf
from .line_search import backtracking_line_search


def check_col(p, obs):
    return sdf.min_sdf_path(p, obs, n_interp=25) < 0


def collision_check(t, p, direction, obs, beta=0.5, min_step=1e-8):
    while check_col([p, p + t * direction], obs):
        t = t * beta
        if t < min_step:
            return t
    return t


def cap_step(step, direction, max_move=0.15):
    move_norm = np.linalg.norm(step * direction)
    if move_norm > max_move:
        step = max_move / np.linalg.norm(direction)
    return step


def plan(env, t0=0.005, alpha=1e-4, beta=0.5, max_iter=2000, tol=0.05, d0=1.5, max_move=0.15):
    p = env.start.copy()
    path = [p.copy()]

    def f(x):
        return pf.total(x, env, d0=d0)

    def grad(x):
        return pf.total_grad(x, env, d0=d0)

    for i in range(max_iter):
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i

        g = grad(p)

        if np.linalg.norm(g) < 1e-10:
            return np.array(path), False, i

        direction = -g
        step = backtracking_line_search(f, grad, p, direction, t0=t0, alpha=alpha, beta=beta)
        step = collision_check(step, p, direction, env.obstacles, beta=beta)
        step = cap_step(step, direction, max_move=max_move)

        p = p + step * direction
        path.append(p.copy())

    return np.array(path), False, max_iter