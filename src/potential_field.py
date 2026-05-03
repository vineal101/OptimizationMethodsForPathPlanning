import numpy as np


def attractive(p, goal, k_att=1.0):
    return 0.5 * k_att * np.sum((p - goal) ** 2)


def attractive_grad(p, goal, k_att=1.0):
    return k_att * (p - goal)

def vector_bounds(bounds):
    edges = []
    for i, bound in enumerate(bounds[0]):
        edge1 = [bound, bound]
        edge2 = [bound, bounds[0][i-1]]
        edges.append(np.array([edge1, edge2]))
        edge1 = [bound, bound]
        edge2 = [bounds[0][i-1], bound]
        edges.append(np.array([edge1, edge2]))
    return edges

def bound_dist(p, e1, e2):
    e1p = p - e1
    e1e2 = e2 - e1
    t = np.dot(e1p, e1e2) / np.dot(e1e2, e1e2)
    t = np.clip(t, 0, 1)
    return e1 + t*e1e2

def repulsive(p, obstacles, bounds, k_rep=100.0, d0=1.5, bound_d=0.25):
    u = 0.0
    for o in obstacles:
        d = np.linalg.norm(p - o["center"]) - o["radius"]
        if 0 < d < d0:
            u += 0.5 * k_rep * (1.0 / d - 1.0 / d0) ** 2
        elif d <= 0:
            u += 1e6
    edges = vector_bounds(bounds)
    for edge in edges:
        dist = np.linalg.norm(p - bound_dist(p, edge[0], edge[1]))
        if 0 < dist < bound_d:
            u += 0.5 * k_rep * (1.0 / dist - 1.0 / bound_d) ** 2
    return u


def repulsive_grad(p, obstacles, bounds, k_rep=100.0, d0=1.5, bound_d=0.25):
    g = np.zeros_like(p)
    for o in obstacles:
        diff = p - o["center"]
        dist = np.linalg.norm(diff)
        d = dist - o["radius"]
        if 0 < d < d0 and dist > 1e-9:
            g += -k_rep * (1.0 / d - 1.0 / d0) * (1.0 / d ** 2) * (diff / dist)
    edges = vector_bounds(bounds)
    for edge in edges:
        diff = p - bound_dist(p, edge[0], edge[1])
        dist = np.linalg.norm(diff)
        if 0 < dist < bound_d:
            g += -k_rep * (1.0 / dist - 1.0 / bound_d) * (1.0 / dist ** 2) * (diff / dist)
    return g


def total(p, env, k_att=10.0, k_rep=100.0, d0=1.5, bound_d=0.25):
    return attractive(p, env.goal, k_att) + repulsive(p, env.obstacles, env.bounds, k_rep, d0, bound_d)


def total_grad(p, env, k_att=10.0, k_rep=100.0, d0=1.5, bound_d=0.25):
    return attractive_grad(p, env.goal, k_att) + repulsive_grad(p, env.obstacles, env.bounds, k_rep, d0, bound_d)

def repulsive_hess(p, obstacles, bounds, k_rep=100.0, d0=1.5, bound_d=0.25):
    H = np.zeros((len(p),len(p)))
    for o in obstacles:
        diff = p - o["center"]
        dist = np.linalg.norm(diff)
        d = dist - o["radius"]
        dir = diff/dist
        if 0 < d < d0 and dist > 1e-9:
            H_d = (np.identity(2) - np.outer(dir, dir)) / dist
            pf_prime = -k_rep * (1.0/d - 1.0/d0) / (d ** 2)
            pf_double = k_rep * (3.0/(d ** 4) - 2.0/(d0 * (d ** 3)))
            H += pf_double * np.outer(dir, dir) + pf_prime * H_d
    edges = vector_bounds(bounds)
    for edge in edges:
        diff = p - bound_dist(p, edge[0], edge[1])
        dist = np.linalg.norm(diff)
        dir = diff/dist
        if 0 < dist < bound_d:
            H_d = (np.identity(2) - np.outer(dir, dir)) / dist
            pf_prime = -k_rep * (1.0/dist - 1.0/bound_d) / (dist ** 2)
            pf_double = k_rep * (3.0/(dist ** 4) - 2.0/(bound_d * (dist ** 3)))
            H += pf_double * np.outer(dir, dir) + pf_prime * H_d
    return H

def attractive_hess(p, k_att=1.0):
    return k_att * np.identity(len(p))

def hessian(p, env, k_att=1.0, k_rep=100.0, d0=1.5):
    return repulsive_hess(p, env.obstacles, env.bounds, k_rep, d0) + attractive_hess(p, k_att)