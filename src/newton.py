import numpy as np
from . import potential_field as pf
from . import sdf
from .line_search import backtracking_newton
import sys

def check_col(p, obs):
    if sdf.min_sdf_path(p, obs, n_interp=25) < 0:
        return True
    else:
        return False
    
def collision_check(t, p, dir, obs, beta=0.5):
    while(check_col([p, p + t*dir], obs)):
        t = t * beta
    return t
    
def plan(env, alpha=10**-4, beta=0.5, max_iter=2000, tol=0.05, d0=1.5, t0=1.0):
    p = env.start.copy()
    path = [p.copy()]

    def f(x):
        return pf.total(x, env, d0=d0)
    
    def grad(x):
        return pf.total_grad(x, env, d0=d0)
    
    for i in range(max_iter):
        H = pf.hessian(p, env, d0=d0)
        eigvals = np.linalg.eigvals(H)
        lambda_min = np.min(eigvals)
        # if H has any negative eigenvalues force it PD
        if lambda_min <= 0:
            H += (abs(lambda_min) + 1e-3) * np.identity(2)
        g = grad(p)
        # check if hessian is invertible and pd
        if np.linalg.det(H) == 0:
            sys.exit("hessian not invertible")
            break
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            sys.exit("hessian not pos def")
        dir = np.linalg.solve(H, (-1)*g)
        decrement = np.sqrt(g.T @ np.linalg.inv(H) @ g)
        t = backtracking_newton(f, p, dir, decrement, t0, alpha=alpha, beta=beta)
        t = collision_check(t, p, dir, env.obstacles)
        p = p + t * dir
        path.append(p.copy())
        if np.linalg.norm(p - env.goal) < tol:
            return np.array(path), True, i
    return np.array(path), False, max_iter