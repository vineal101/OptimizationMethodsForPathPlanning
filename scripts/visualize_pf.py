import sys
import os 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import src.potential_field as pf
from src.map_env import MapEnv

MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "maps")

def main():
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

    for m in maps:
        env_name = m.replace(".json", "")
        env = MapEnv.from_json(os.path.join(MAPS_DIR, m))
        x = np.outer(np.linspace(0.01, 9.99, 150), np.ones(150))
        y = x.copy().T

        name = env_name + "_vis"

        pot = np.zeros((150,150))
        for i, row in enumerate(pot):
            for j, col in enumerate(row):
                pot[i][j] = pf.total([x[i][j],y[i][j]], env)
                # print(pot[i][j])
                # # pot[i,j] = np.clip(pot[i,j], 0, 1)
                if pot[i][j] >=80:
                    pot[i][j] = 80
        # print(np.min(pot))
        # print(np.max(pot))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.plot_surface(x, y, pot, cmap='viridis', edgecolor='green')
        ax.set_title('Potential Field Plot of ' + env_name)
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=120)
        plt.close(fig)

if __name__ == "__main__":
    main()