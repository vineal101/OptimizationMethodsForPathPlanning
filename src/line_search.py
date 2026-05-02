import numpy as np

def backtracking_line_search(f, grad, x, direction, t0=1.0, alpha=1e-4, beta=0.5, min_step=1e-8, max_backtracks=50):
    t = t0
    fx = f(x)
    gx = grad(x)
    directional_derivative = gx @ direction

    for _ in range(max_backtracks):
        if f(x + t * direction) <= fx + alpha * t * directional_derivative:
            return t
        t *= beta
        if t < min_step:
            return t
    return t