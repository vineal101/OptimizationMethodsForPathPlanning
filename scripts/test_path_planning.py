import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map_env import MapEnv
from src.gradient_descent import plan as gd_plan
from src.accelerated_gradient_descent import plan as agd_plan
from src.newton import plan as newton_plan
from src.bfgs import plan as bfgs_plan
from src.metrics import evaluate_path

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(OUT_DIR, "path_plots")
ALM_HISTORY_DIR = os.path.join(OUT_DIR, "alm_histories")

DEFAULT_MAPS = [
    "easy_1.json",
    "easy_2.json",
    "easy_3.json",
    "medium_1.json",
    "medium_2.json",
    "medium_3.json",
    "hard_1.json",
    "hard_2.json",
    "hard_3.json",
]

PF_KWARGS = {
    "t0": 0.015,
    "alpha": 1e-4,
    "beta": 0.5,
    "max_iter": 2000,
    "tol": 0.05,
    "d0": 1.0,
}


def available_maps():
    maps = [m for m in DEFAULT_MAPS if os.path.exists(os.path.join(MAPS_DIR, m))]
    if maps:
        return maps
    return sorted(m for m in os.listdir(MAPS_DIR) if m.endswith(".json"))


def plot_path(env, path, success, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o["center"], o["radius"], color="gray", alpha=0.7))
    ax.plot(path[:, 0], path[:, 1], "-o", lw=1.5, ms=2)
    ax.plot(*env.start, "go", label="start")
    ax.plot(*env.goal, "r*", ms=12, label="goal")
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_aspect("equal")
    ax.set_title(f"{name}  success={success}  steps={len(path)}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), dpi=120)
    plt.close(fig)


def run_pf_planner(plan, env, include_max_move=False):
    kwargs = dict(PF_KWARGS)
    if include_max_move:
        kwargs["max_move"] = 1000
    return plan(env, **kwargs)


def run_alm_planner(env):
    from src.alm import plan as alm_plan
    return alm_plan(env, seed=0)


def write_alm_history(env_name, history):
    if not history:
        return
    path = os.path.join(ALM_HISTORY_DIR, f"{env_name}_alm_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def empty_metric_row(map_name, planner_name, wall_time_s, error):
    return {
        "map": map_name,
        "difficulty": map_name.split("_")[0],
        "planner": planner_name,
        "success": False,
        "n_iters": -1,
        "wall_time_s": wall_time_s,
        "path_distance": math.nan,
        "path_smoothness": math.nan,
        "min_sdf": math.nan,
        "goal_error": math.nan,
        "final_rho": math.nan,
        "final_violation": math.nan,
        "n_active": math.nan,
        "avg_inner": math.nan,
        "error": error,
    }


def run_one(env, env_name, planner_name, runner):
    start_time = time.perf_counter()
    history = None

    try:
        result = runner(env)
        wall_time_s = time.perf_counter() - start_time

        if planner_name == "alm":
            path, success, iters, history = result
        else:
            path, success, iters = result

        metrics = evaluate_path(path, env, wall_time_s, iters, success)
        row = {
            "map": env_name,
            "difficulty": env_name.split("_")[0],
            "planner": planner_name,
            **metrics,
            "final_rho": math.nan,
            "final_violation": math.nan,
            "n_active": math.nan,
            "avg_inner": math.nan,
            "error": "",
        }

        if planner_name == "alm" and history:
            final = history[-1]
            row.update(
                {
                    "final_rho": final.get("rho", math.nan),
                    "final_violation": final.get("violation", math.nan),
                    "n_active": final.get("n_active", math.nan),
                    "avg_inner": float(np.mean([h["inner_iters"] for h in history])),
                }
            )
            write_alm_history(env_name, history)

        plot_path(env, path, success, f"{env_name}_{planner_name}")
        return row

    except SystemExit as e:
        wall_time_s = time.perf_counter() - start_time
        return empty_metric_row(env_name, planner_name, wall_time_s, f"SystemExit: {e}")
    except Exception as e:
        wall_time_s = time.perf_counter() - start_time
        return empty_metric_row(env_name, planner_name, wall_time_s, repr(e))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_numeric(rows, key):
    vals = []
    for row in rows:
        val = row.get(key, math.nan)
        if isinstance(val, bool):
            val = float(val)
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isnan(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else math.nan


def build_summary(rows):
    summary = []
    groups = sorted({(row["planner"], row["difficulty"]) for row in rows})

    for planner, difficulty in groups:
        group = [row for row in rows if row["planner"] == planner and row["difficulty"] == difficulty]
        summary.append(
            {
                "planner": planner,
                "difficulty": difficulty,
                "num_runs": len(group),
                "success_rate": mean_numeric(group, "success"),
                "avg_iters": mean_numeric(group, "n_iters"),
                "avg_time_s": mean_numeric(group, "wall_time_s"),
                "avg_path_distance": mean_numeric(group, "path_distance"),
                "avg_smoothness": mean_numeric(group, "path_smoothness"),
                "avg_min_sdf": mean_numeric(group, "min_sdf"),
                "avg_goal_error": mean_numeric(group, "goal_error"),
            }
        )

    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ALM_HISTORY_DIR, exist_ok=True)

    planners = {
        "gd": lambda env: run_pf_planner(gd_plan, env, include_max_move=True),
        "agd": lambda env: run_pf_planner(agd_plan, env, include_max_move=True),
        "newton": lambda env: run_pf_planner(newton_plan, env),
        "bfgs": lambda env: run_pf_planner(bfgs_plan, env),
        "alm": run_alm_planner,
    }

    rows = []

    for m in available_maps():
        env_name = m.replace(".json", "")
        env = MapEnv.from_json(os.path.join(MAPS_DIR, m))

        for planner_name, runner in planners.items():
            row = run_one(env, env_name, planner_name, runner)
            rows.append(row)

            print(
                f"{env_name}_{planner_name}: "
                f"success={row['success']}, "
                f"n_iters={row['n_iters']}, "
                f"wall_time_s={row['wall_time_s']:.6f}, "
                f"path_distance={row['path_distance']:.4f}, "
                f"path_smoothness={row['path_smoothness']:.4f}, "
                f"min_sdf={row['min_sdf']:.4f}, "
                f"goal_error={row['goal_error']:.4f}"
                + (f", error={row['error']}" if row["error"] else "")
            )

    write_csv(os.path.join(OUT_DIR, "all_methods_metrics.csv"), rows)
    write_csv(os.path.join(OUT_DIR, "all_methods_summary_by_difficulty.csv"), build_summary(rows))

    print(f"\nSaved raw metrics to {os.path.join(OUT_DIR, 'all_methods_metrics.csv')}")
    print(f"Saved summary table to {os.path.join(OUT_DIR, 'all_methods_summary_by_difficulty.csv')}")
    print(f"Saved path plots to {PLOTS_DIR}")
    print(f"Saved ALM histories to {ALM_HISTORY_DIR}")


if __name__ == "__main__":
    main()
