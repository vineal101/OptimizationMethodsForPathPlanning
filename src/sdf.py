# src/sdf.py
import numpy as np


def sdf_circle(p, center, radius):
    return np.linalg.norm(p - center) - radius


def sdf_circle_grad(p, center, radius):
    diff = p - center
    dist = np.linalg.norm(diff)
    if dist < 1e-9:
        return np.zeros_like(p)
    return diff / dist


def min_sdf_path(path, obstacles, n_interp=10):
    min_d = np.inf
    for i in range(len(path) - 1):
        for t in np.linspace(0, 1, n_interp):
            p = (1 - t) * np.array(path[i]) + t * np.array(path[i + 1])
            for o in obstacles:
                d = sdf_circle(p, o["center"], o["radius"])
                min_d = min(min_d, d)
    return min_d
