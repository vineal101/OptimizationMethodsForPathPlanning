import numpy as np
from . import potential_field as pf
from . import sdf
import sys

def check_col(p, obs):
    if sdf.min_sdf_path(p, obs, n_interp=25) < 0:
        return True
    else:
        return False
    
def plan(env, alpha=10**-4, beta=0.5, max_iter=2000, tol=0.05):
    p = env.start.copy()
    path = [p.copy()]
    for i in range(max_iter):
        H = pf.hessian(p, env.obstacles)
        H_reg = 1e-6*np.eye(2)
        g = pf.total_grad(p, env.goal, env.obstacles)
        # check if hessian is invertible and pd
        if np.linalg.det(H_reg) == 0:
            sys.exit("hessian not invertible")
            break
        try:
            np.linalg.cholesky(H_reg)
        except np.linalg.LinAlgError:
            sys.exit("hessian not pos def")
        dir = np.linalg.solve(H_reg, (-1)*g)
        decrement = np.sqrt(g.T @ np.linalg.inv(H_reg) @ g)
        t = 1
        while(pf.total(p+t*dir, env.goal, env.obstacles)>pf.total(p, env.goal, env.obstacles) + alpha*t*(decrement ** 2) or check_col([p, p+t*dir], env.obstacles)):
            t = t * beta
        p = p + t * dir
        path.append(p.copy())
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
    return np.array(path), False, max_iter