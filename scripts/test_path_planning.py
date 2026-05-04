import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map_env import MapEnv
from src.gradient_descent import plan as gd_plan
from src.accelerated_gradient_descent import plan as agd_plan

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

    maps = ["easy_open.json", "easy_one_obstacle.json", "easy_two_obstacles.json"]

    planners = {
        "gd": gd_plan,
        "agd": agd_plan,
    }

    for m in maps:
        env_name = m.replace(".json", "")
        env = MapEnv.from_json(os.path.join(MAPS_DIR, m))

        for planner_name, plan in planners.items():
            name = f"{env_name}_{planner_name}"
            path, success, iters = plan(env, t0=0.01, alpha=1e-4, beta=0.5, max_iter=2000, tol=0.05, d0=1.5, max_move=1000)

            print(f"{name}: success={success}, iters={iters}, end={path[-1]}")
            plot(env, path, success, name)


if __name__ == "__main__":
    main()