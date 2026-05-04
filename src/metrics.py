import numpy as np

from src.config import load as load_config
from src import sdf

CFG = load_config()


def evaluate_path(path, env, wall_time_s, n_iters, success):
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    second_diffs = path[2:] - 2 * path[1:-1] + path[:-2]
    return {
        "success": bool(success),
        "n_iters": int(n_iters),
        "wall_time_s": float(wall_time_s),
        "path_distance": float(np.sum(seg_lengths)),
        "path_smoothness": float(np.sum(np.linalg.norm(second_diffs, axis=1) ** 2)),
        "min_sdf": float(
            sdf.min_sdf_path(
                path, env.obstacles, n_interp=CFG["shared"]["collision_n_interp"]
            )
        ),
        "goal_error": float(np.linalg.norm(path[-1] - env.goal)),
    }
