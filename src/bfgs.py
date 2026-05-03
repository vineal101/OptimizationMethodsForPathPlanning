import numpy as np
from . import potential_field as pf
from . import sdf
from .line_search import backtracking_line_search 

def check_col(p, obs):
    if sdf.min_sdf_path(p, obs, n_interp=100) < 0:
        return True
    else:
        return False

def collision_check(t, p, dir, obs, beta=0.5):
    while(check_col([p, p + t*dir], obs)):
        t = t * beta
    return t

def next_b(y, del_x, B_inv, denom):
    return (np.identity(2)-(del_x @ y.T)/(denom)) @ B_inv @ (np.identity(2) - (y @ del_x.T)/(denom)) + (del_x @ del_x.T)/denom

def plan(env, t0=1.0, alpha=10**-4, beta=0.5, max_iter=2000, tol=0.05, d0=1.5):
    p = env.start.copy()
    path = [p.copy()]
    B_inv = np.identity(2)
    f_prev = pf.total_grad(p, env)
    f_next = 0

    def f(x):
        return pf.total(x, env, d0=d0)
    
    def grad(x):
        return pf.total_grad(x, env, d0=d0)
    
    for i in range(max_iter):
        dir = -B_inv @ f_prev
        t = backtracking_line_search(f, grad, p, dir, alpha=alpha, beta=beta, t0=t0)
        t = collision_check(t, p, dir, env.obstacles)
        p = p + t * dir
        path.append(p.copy())
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
        f_next = grad(p)
        y = f_next - f_prev
        f_prev = f_next
        del_x = p - path[i]
        denom = y.T @ del_x
        B_inv = next_b(y, del_x, B_inv, denom)
    return np.array(path), False, max_iter