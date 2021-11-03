import json
from typing import Dict, List


def save_json(samples: List[Dict], path: str):
    with open(path, "w") as jf:
        json.dump(samples, jf)


def load_json(path: str):
    with open(path, "r") as jf:
        return json.load(jf)
