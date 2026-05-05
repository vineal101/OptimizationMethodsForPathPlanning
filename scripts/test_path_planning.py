import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map_env import MapEnv
from src.gradient_descent import plan as gd_plan
from src.accelerated_gradient_descent import plan as agd_plan
from src.metrics import evaluate_path

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def plot(env, path, success, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o["center"], o["radius"], color="gray"))
    ax.plot(path[:, 0], path[:, 1], "-b", lw=1)
    ax.plot(*env.start, "go", label="start")
    ax.plot(*env.goal, "r*", ms=12, label="goal")
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_aspect("equal")
    ax.set_title(f"{name}  success={success}  steps={len(path)}")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    maps = [
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

    planners = {
        "gd": gd_plan,
        "agd": agd_plan,
    }

    rows = []

    for m in maps:
        env_name = m.replace(".json", "")
        env = MapEnv.from_json(os.path.join(MAPS_DIR, m))

        for planner_name, plan in planners.items():
            name = f"{env_name}_{planner_name}"

            start_time = time.perf_counter()
            path, success, iters = plan(env, t0=0.015, alpha=1e-4, beta=0.5, max_iter=2000, tol=0.05, d0=1.0, max_move=1000)
            wall_time_s = time.perf_counter() - start_time

            metrics = evaluate_path(path, env, wall_time_s, iters, success)

            row = {
                "map": env_name,
                "planner": planner_name,
                **metrics,
            }

            rows.append(row)

            print(
                f"{name}: "
                f"success={metrics['success']}, "
                f"n_iters={metrics['n_iters']}, "
                f"wall_time_s={metrics['wall_time_s']:.6f}, "
                f"path_distance={metrics['path_distance']:.4f}, "
                f"path_smoothness={metrics['path_smoothness']:.4f}, "
                f"min_sdf={metrics['min_sdf']:.4f}, "
                f"goal_error={metrics['goal_error']:.4f}"
            )

            plot(env, path, success, name)

    csv_path = os.path.join(OUT_DIR, "gd_agd_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()