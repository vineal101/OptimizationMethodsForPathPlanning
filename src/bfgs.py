import numpy as np
from . import potential_field as pf

def plan(env, alpha=10**-4, beta=0.5, max_iter=2000, tol=0.05):
    p = env.start.copy()
    path = [p.copy()]
    B_inv = np.identity(2)
    f_prev = pf.total_grad(p, env.goal, env.obstacles, d0=2)
    f_next = 0
    for i in range(max_iter):
        dir = -B_inv @ f_prev
        t = 0.3
        while(pf.total(p+t*dir, env.goal, env.obstacles, d0=2)>pf.total(p, env.goal, env.obstacles, d0=2) + alpha*t*(f_prev @ dir)):
            t = t * beta
        p = p + t * dir
        path.append(p.copy())
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
        f_next = pf.total_grad(p, env.goal, env.obstacles, d0=2)
        y = f_next - f_prev
        f_prev = f_next
        del_x = p - path[i]
        denom = y.T @ del_x
        B_inv = (np.identity(2)-(del_x @ y.T)/(denom)) @ B_inv @ (np.identity(2) - (y @ del_x.T)/(denom)) + (del_x @ del_x.T)/denom
