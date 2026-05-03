import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.map_env import MapEnv
from src import sdf
from src import trajectory_costs as tc


def build_subproblem(env, T, d_safe, alpha, beta):
    s, goal = env.start.astype(float), env.goal.astype(float)
    n_obs = len(env.obstacles)
    n_con = (T - 1) * n_obs

    def reconstruct(x_flat):
        return np.vstack([s, x_flat.reshape(T - 1, 2), goal])

    def constraints(p):
        c = np.empty(n_con)
        i = 0
        for t in range(1, T):
            for o in env.obstacles:
                c[i] = d_safe - sdf.sdf_circle(p[t], o["center"], o["radius"])
                i += 1
        return c

    def L_A(x_flat, lam, rho):
        p = reconstruct(x_flat)
        f = alpha * tc.smoothness(p) + beta * tc.length(p)
        active = np.maximum(lam + rho * constraints(p), 0.0)
        return f + np.sum(active**2 - lam**2) / (2.0 * rho)

    def L_A_grad(x_flat, lam, rho):
        p = reconstruct(x_flat)
        grad_f = alpha * tc.smoothness_grad(p) + beta * tc.length_grad(p)
        g = grad_f[1:-1].copy()
        i = 0
        for t in range(1, T):
            for o in env.obstacles:
                g_val = d_safe - sdf.sdf_circle(p[t], o["center"], o["radius"])
                a = max(lam[i] + rho * g_val, 0.0)
                if a > 0:
                    g[t - 1] -= a * sdf.sdf_circle_grad(p[t], o["center"], o["radius"])
                i += 1
        return g.flatten()

    return L_A, L_A_grad, constraints, reconstruct, n_con


def plan(
    env,
    T=30,
    d_safe=0.05,
    alpha=1.0,
    beta=10.0,
    rho0=1.0,
    rho_growth=2.0,
    max_outer=50,
    tol_violation=1e-3,
    schedule="adaptive",
    seed=0,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    s, goal = env.start.astype(float), env.goal.astype(float)

    path_scale = np.linalg.norm(goal - s)
    path = np.linspace(s, goal, T + 1)
    path[1:-1] += 0.01 * path_scale / T * rng.standard_normal((T - 1, 2))

    L_A, L_A_grad, constraints, reconstruct, n_con = build_subproblem(
        env, T, d_safe, alpha, beta
    )

    lam = np.zeros(n_con)
    rho = rho0
    interior = path[1:-1].flatten()
    prev_violation = np.inf

    for k in range(max_outer):
        result = minimize(
            lambda x: L_A(x, lam, rho),
            interior,
            jac=lambda x: L_A_grad(x, lam, rho),
            method="L-BFGS-B",
            options={"maxiter": 200, "gtol": 1e-6},
        )
        interior = result.x
        path = reconstruct(interior)

        g_vals = constraints(path)
        violation = float(np.max(np.maximum(g_vals, 0.0)))
        min_sdf = float(sdf.min_sdf_path(path, env.obstacles))

        if verbose:
            print(
                f"  k={k:2d}  viol={violation:.2e}  min_sdf={min_sdf:+.3f}  "
                f"rho={rho:.2f}  inner={result.nit}"
            )

        if violation < tol_violation:
            return path, min_sdf > -1e-3, k

        lam = np.maximum(lam + rho * g_vals, 0.0)

        if schedule == "geometric":
            rho *= rho_growth
        elif schedule == "adaptive":
            if violation > 0.25 * prev_violation:
                rho *= rho_growth
        elif schedule != "fixed":
            raise ValueError(f"unknown schedule: {schedule}")
        prev_violation = violation

    return path, (violation < tol_violation and min_sdf > -1e-3), max_outer


def main():
    out_dir = "results/alm"
    os.makedirs(out_dir, exist_ok=True)

    env = MapEnv.from_json("maps/easy_one_obstacle.json")
    path, success, n_outer = plan(env, verbose=True)

    print(f"\nsuccess={success}, outer iterations={n_outer}")

    fig, ax = plt.subplots(figsize=(5, 5))
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o["center"], o["radius"], color="gray", alpha=0.7))
    ax.plot(path[:, 0], path[:, 1], "-o", lw=2, ms=3, color="C0")
    ax.plot(*env.start, "go", ms=10, label="start")
    ax.plot(*env.goal, "r*", ms=14, label="goal")
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_aspect("equal")
    ax.set_title(f"ALM on easy_one_obstacle  success={success}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/output.png")
    plt.close(fig)

    print(f"saved {out_dir}/output.png")


if __name__ == "__main__":
    main()
