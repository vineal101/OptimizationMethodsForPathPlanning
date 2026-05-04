import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load():
    with open(CONFIG_PATH) as f:
        return json.load(f)
