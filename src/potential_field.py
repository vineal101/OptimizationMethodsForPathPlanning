import numpy as np


def attractive(p, goal, k_att=1.0):
    return 0.5 * k_att * np.sum((p - goal) ** 2)


def attractive_grad(p, goal, k_att=1.0):
    return k_att * (p - goal)


def repulsive(p, obstacles, k_rep=100.0, d0=1.5):
    u = 0.0
    for o in obstacles:
        d = np.linalg.norm(p - o["center"]) - o["radius"]
        if 0 < d < d0:
            u += 0.5 * k_rep * (1.0 / d - 1.0 / d0) ** 2
        elif d <= 0:
            u += 1e6
    return u


def repulsive_grad(p, obstacles, k_rep=100.0, d0=1.5):
    g = np.zeros_like(p)
    for o in obstacles:
        diff = p - o["center"]
        dist = np.linalg.norm(diff)
        d = dist - o["radius"]
        if 0 < d < d0 and dist > 1e-9:
            g += -k_rep * (1.0 / d - 1.0 / d0) * (1.0 / d ** 2) * (diff / dist)
    return g


def total(p, goal, obstacles, k_att=1.0, k_rep=100.0, d0=1.5):
    return attractive(p, goal, k_att) + repulsive(p, obstacles, k_rep, d0)


def total_grad(p, goal, obstacles, k_att=1.0, k_rep=100.0, d0=1.5):
    return attractive_grad(p, goal, k_att) + repulsive_grad(p, obstacles, k_rep, d0)
