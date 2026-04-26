import json
import numpy as np


class MapEnv:
    def __init__(self, bounds, start, goal, obstacles):
        self.bounds = np.asarray(bounds, dtype=float)
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = [
            {"center": np.asarray(o["center"], dtype=float), "radius": float(o["radius"])}
            for o in obstacles
        ]

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["bounds"], d["start"], d["goal"], d["obstacles"])

    def in_collision(self, p):
        for o in self.obstacles:
            if np.linalg.norm(p - o["center"]) <= o["radius"]:
                return True
        return False
