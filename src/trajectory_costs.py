# src/trajectory_costs.py
import numpy as np


def smoothness(path):
    """Sum of squared second differences."""
    diffs = path[2:] - 2 * path[1:-1] + path[:-2]
    return np.sum(diffs**2)


def smoothness_grad(path):
    """Gradient of smoothness w.r.t. all waypoints."""
    n = len(path)
    g = np.zeros_like(path)
    for t in range(1, n - 1):
        d = path[t + 1] - 2 * path[t] + path[t - 1]
        g[t - 1] += 2 * d
        g[t] += -4 * d
        g[t + 1] += 2 * d
    return g


def length(path):
    """Sum of squared segment lengths."""
    diffs = path[1:] - path[:-1]
    return np.sum(diffs**2)


def length_grad(path):
    """Gradient of length w.r.t. all waypoints."""
    n = len(path)
    g = np.zeros_like(path)
    for t in range(n - 1):
        d = path[t + 1] - path[t]
        g[t] += -2 * d
        g[t + 1] += 2 * d
    return g
