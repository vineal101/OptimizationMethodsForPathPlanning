import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.map_env import MapEnv
from src.config import load as load_config
from src import sdf
from src import trajectory_costs as tc
from src.metrics import evaluate_path

CFG = load_config()
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "alm")
MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")

SUB_DIRS = {
    "validation": os.path.join(OUT_DIR, "validation"),
    "paths": os.path.join(OUT_DIR, "paths"),
    "convergence": os.path.join(OUT_DIR, "convergence"),
    "ablation": os.path.join(OUT_DIR, "ablation"),
    "histories": os.path.join(OUT_DIR, "histories"),
}


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


def plan(env, seed=0, verbose=False, **overrides):
    p = {**CFG["alm"], **overrides}
    rng = np.random.default_rng(seed)
    s, goal = env.start.astype(float), env.goal.astype(float)

    path_scale = np.linalg.norm(goal - s)
    path = np.linspace(s, goal, p["T"] + 1)
    path[1:-1] += 0.01 * path_scale / p["T"] * rng.standard_normal((p["T"] - 1, 2))

    L_A, L_A_grad, constraints, reconstruct, n_con = build_subproblem(
        env, p["T"], p["d_safe"], p["alpha"], p["beta"]
    )

    lam = np.zeros(n_con)
    rho = p["rho0"]
    interior = path[1:-1].flatten()
    history = []
    prev_violation = np.inf

    for k in range(p["max_outer"]):
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
        violation = float(np.max(np.maximum(g_vals, 0.0))) if g_vals.size > 0 else 0.0
        min_sdf = float(
            sdf.min_sdf_path(
                path, env.obstacles, n_interp=CFG["shared"]["collision_n_interp"]
            )
        )
        history.append(
            {
                "outer_iter": k,
                "cost": float(result.fun),
                "violation": violation,
                "min_sdf": min_sdf,
                "rho": float(rho),
                "inner_iters": int(result.nit),
                "n_active": int(np.sum(lam > 1e-6)),
            }
        )

        if verbose:
            print(
                f"  k={k:2d}  viol={violation:.2e}  min_sdf={min_sdf:+.3f}  "
                f"rho={rho:.2f}  inner={result.nit}"
            )

        if violation < p["tol_violation"]:
            return path, min_sdf > -1e-3, k, history

        lam = np.maximum(lam + rho * g_vals, 0.0)

        if p["schedule"] == "geometric":
            rho *= p["rho_growth"]
        elif p["schedule"] == "adaptive":
            if violation > 0.25 * prev_violation:
                rho *= p["rho_growth"]
        elif p["schedule"] != "fixed":
            raise ValueError(f"unknown schedule: {p['schedule']}")
        prev_violation = violation

    success = (violation < p["tol_violation"]) and (min_sdf > -1e-3)
    return path, success, p["max_outer"], history


def validate_qp():
    print("Validation: QP vs CVXPY")
    rng = np.random.default_rng(42)
    n, m = 10, 3
    M = rng.standard_normal((n, n))
    Q = M.T @ M + np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)

    x = cp.Variable(n)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x), [A @ x == b]).solve()
    x_star = x.value

    lam = np.zeros(m)
    rho = 1.0
    hist = []
    for k in range(80):
        x_alm = -np.linalg.solve(Q + rho * A.T @ A, c + A.T @ lam - rho * A.T @ b)
        viol = np.linalg.norm(A @ x_alm - b)
        dist = np.linalg.norm(x_alm - x_star)
        hist.append((k, viol, dist))
        if viol < 1e-12:
            break
        lam += rho * (A @ x_alm - b)

    err = np.linalg.norm(x_alm - x_star)
    print(f"  |x_alm - x_star| = {err:.2e}")
    assert err < 1e-8, "QP doesn't match CVXPY"

    valid = [(k, d) for (k, _, d) in hist if d > 1e-14]
    ks = np.array([v[0] for v in valid])
    ds = np.array([v[1] for v in valid])
    slope, intercept = np.polyfit(ks, np.log(ds), 1)
    rate = np.exp(slope)
    print(f"  empirical contraction factor: {rate:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ks, ds, "o-", label=r"empirical $\|x^k - x^*\|$")
    ax.semilogy(
        ks,
        np.exp(intercept + slope * ks),
        "--",
        label=f"linear fit (rate = {rate:.3f}/iter)",
    )
    ax.set_xlabel("Outer iteration k")
    ax.set_ylabel("Distance to optimum")
    ax.set_title("ALM linear convergence on convex QP")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['validation']}/qp_convergence.png")
    plt.close(fig)


def plot_path(env, path, name, ok):
    fig, ax = plt.subplots(figsize=(5, 5))
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o["center"], o["radius"], color="gray", alpha=0.7))
    ax.plot(path[:, 0], path[:, 1], "-o", lw=2, ms=3, color="C0")
    ax.plot(*env.start, "go", ms=10, label="start")
    ax.plot(*env.goal, "r*", ms=14, label="goal")
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_aspect("equal")
    ax.set_title(f"ALM on {name}  success={ok}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['paths']}/{name}_path.png")
    plt.close(fig)


def plot_convergence(history, name):
    ks = [h["outer_iter"] for h in history]
    viols = [max(h["violation"], 1e-12) for h in history]
    rhos = [h["rho"] for h in history]
    inners = [h["inner_iters"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(ks, viols, "o-")
    axes[0].set(
        xlabel="Outer iteration",
        ylabel="Max violation",
        title=f"{name}: constraint violation",
    )
    axes[1].plot(ks, rhos, "o-")
    axes[1].set(
        xlabel="Outer iteration", ylabel=r"$\rho$", title=f"{name}: penalty parameter"
    )
    axes[2].plot(ks, inners, "o-")
    axes[2].set(
        xlabel="Outer iteration",
        ylabel="Inner L-BFGS-B iterations",
        title=f"{name}: inner iterations",
    )
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['convergence']}/{name}_convergence.png")
    plt.close(fig)


def run_experiments():
    print("\nMain experiments")
    summary = []
    for name in CFG["shared"]["maps"]:
        env = MapEnv.from_json(os.path.join(MAPS_DIR, f"{name}.json"))
        for seed in CFG["shared"]["seeds"]:
            t0 = time.perf_counter()
            path, ok, n_outer, history = plan(env, seed=seed)
            wall_time = time.perf_counter() - t0

            metrics = evaluate_path(path, env, wall_time, n_outer, ok)
            final = history[-1]
            summary.append(
                {
                    "method": "alm",
                    "map": name,
                    "seed": seed,
                    **metrics,
                    "final_rho": final["rho"],
                    "n_active": final["n_active"],
                    "avg_inner": float(np.mean([h["inner_iters"] for h in history])),
                }
            )

            print(
                f"  {name} seed={seed}: ok={ok}  outer={n_outer}  "
                f"min_sdf={metrics['min_sdf']:+.3f}  "
                f"time={wall_time:.2f}s  active={final['n_active']}"
            )

            if seed == 0:
                plot_path(env, path, name, ok)
                plot_convergence(history, name)
                with open(
                    f"{SUB_DIRS['histories']}/{name}_history.csv", "w", newline=""
                ) as fp:
                    w = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
                    w.writeheader()
                    w.writerows(history)

    with open(f"{OUT_DIR}/summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)


def run_ablation():
    print("\nPenalty schedule ablation")
    env = MapEnv.from_json(os.path.join(MAPS_DIR, "easy_one_obstacle.json"))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"fixed": "C0", "geometric": "C1", "adaptive": "C2"}

    for sched in ["fixed", "geometric", "adaptive"]:
        _, _, n_outer, history = plan(env, seed=0, schedule=sched)
        ks = [h["outer_iter"] for h in history]
        viols = [max(h["violation"], 1e-12) for h in history]
        ax.semilogy(
            ks,
            viols,
            "o-",
            color=colors[sched],
            label=f"{sched}  (final = {viols[-1]:.1e})",
        )
        print(f"  {sched}: final violation = {viols[-1]:.1e}, outer iters = {n_outer}")

    ax.set(
        xlabel="Outer iteration",
        ylabel="Max constraint violation (log)",
        title="Penalty schedule ablation: easy_one_obstacle",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['ablation']}/rho_ablation.png")
    plt.close(fig)


def main():
    for d in SUB_DIRS.values():
        os.makedirs(d, exist_ok=True)
    validate_qp()
    run_experiments()
    run_ablation()
    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
