import json
import math
import os
import random
from collections import deque


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPS_DIR = os.path.join(PROJECT_ROOT, "maps")

BOUNDS = [[0, 10], [0, 10]]
START = [1, 1]
GOAL = [9, 9]

DIFFICULTIES = {
    "easy": {"coverage": 0.02, "rmin": 0.3, "rmax": 1, "count": 1},
    "medium": {"coverage": 0.04, "rmin": 0.3, "rmax": 1, "count": 1},
    "hard": {"coverage": 0.08, "rmin": 0.3, "rmax": 1, "count": 1},
}

START_GOAL_CLEARANCE = 1.0
OBSTACLE_GAP = 0.05
GRID_SIZE = 120


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_area():
    return (BOUNDS[0][1] - BOUNDS[0][0]) * (BOUNDS[1][1] - BOUNDS[1][0])


def obstacle_area(obstacles):
    return sum(math.pi * o["radius"] ** 2 for o in obstacles)


def collides_with_point(circle, point, clearance):
    return dist(circle["center"], point) <= circle["radius"] + clearance


def overlaps(circle, obstacles):
    for o in obstacles:
        if dist(circle["center"], o["center"]) < circle["radius"] + o["radius"] + OBSTACLE_GAP:
            return True
    return False


def sample_circle(rmin, rmax):
    r = random.uniform(rmin, rmax)
    x = random.uniform(BOUNDS[0][0] + r, BOUNDS[0][1] - r)
    y = random.uniform(BOUNDS[1][0] + r, BOUNDS[1][1] - r)
    return {"center": [x, y], "radius": r}


def point_free(point, obstacles, clearance=0.0):
    for o in obstacles:
        if dist(point, o["center"]) <= o["radius"] + clearance:
            return False
    return True


def cell_to_world(cell):
    x = BOUNDS[0][0] + cell[0] / (GRID_SIZE - 1) * (BOUNDS[0][1] - BOUNDS[0][0])
    y = BOUNDS[1][0] + cell[1] / (GRID_SIZE - 1) * (BOUNDS[1][1] - BOUNDS[1][0])
    return [x, y]


def point_to_cell(point):
    x = round((point[0] - BOUNDS[0][0]) / (BOUNDS[0][1] - BOUNDS[0][0]) * (GRID_SIZE - 1))
    y = round((point[1] - BOUNDS[1][0]) / (BOUNDS[1][1] - BOUNDS[1][0]) * (GRID_SIZE - 1))
    return int(x), int(y)


def path_exists(obstacles):
    return True


def generate_obstacles(config):
    obstacles = []
    target_area = config["coverage"] * map_area()
    attempts = 0

    while obstacle_area(obstacles) < target_area and attempts < 10000:
        attempts += 1
        circle = sample_circle(config["rmin"], config["rmax"])

        if collides_with_point(circle, START, START_GOAL_CLEARANCE):
            continue

        if collides_with_point(circle, GOAL, START_GOAL_CLEARANCE):
            continue

        if overlaps(circle, obstacles):
            continue

        obstacles.append(circle)

    return obstacles


def generate_map(difficulty, index):
    config = DIFFICULTIES[difficulty]

    for _ in range(200):
        obstacles = generate_obstacles(config)

        if path_exists(obstacles):
            return {
                "name": f"{difficulty}_{index}",
                "description": f"Generated {difficulty} map.",
                "bounds": BOUNDS,
                "start": START,
                "goal": GOAL,
                "obstacles": obstacles,
            }

def main():
    os.makedirs(MAPS_DIR, exist_ok=True)

    for difficulty, config in DIFFICULTIES.items():
        for i in range(1, config["count"] + 1):
            data = generate_map(difficulty, i)
            path = os.path.join(MAPS_DIR, f"{difficulty}_{i}.json")

            with open(path, "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()