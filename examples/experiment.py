import argparse
import os
import sys
import traceback
import warnings
from typing import Dict, Optional, Tuple

import requests

from examples.util import DEFAULT_EXTRA_VARIABLES, get_dir_from_task
from trapper.training.train import run_experiment

__arguments__ = ["config", "task", "experiment_name"]


def download_squad(
    task: str, version: str = "1.1", overwrite: bool = False
) -> Tuple[str, str]:
    """
    Downloads SQuAD dataset with given version.

    Args:
        version: SQuAD dataset version.
        overwrite: If true, overwrites the destination file.

    Returns: (train set path, dev set path) local paths of downloaded dataset files.

    """
    destination_dir = DATASET_DIR.format(task=task)
    os.makedirs(destination_dir, exist_ok=True)
    dataset_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    train_set = f"train-v{version}.json"
    dev_set = f"dev-v{version}.json"

    datasets = [train_set, dev_set]
    paths = []

    for dataset in datasets:
        dest_name = "train.json" if "train" in dataset else "dev.json"
        url = os.path.join(dataset_base_url, dataset)
        dest = os.path.join(destination_dir, dest_name)
        paths.append(dest)

        if not overwrite and os.path.exists(dest):
            warnings.warn(f"{dest} already exists, not overwriting.")
            continue

        r = requests.get(url, allow_redirects=True)

        with open(dest, "wb") as out_file:
            out_file.write(r.content)

    return paths[0], paths[1]


def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment.jsonnet"
    )
    parser.add_argument("--task", type=str, required=True, help="Name of the task")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Handle unset arguments
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split("=")[0])

    return parser.parse_args()


def get_extra_variables(args):
    ext_vars = {}
    for arg, val in args.__dict__.items():
        if arg not in __arguments__:
            ext_vars[arg.upper()] = val
    return ext_vars


def validate_extra_variables(
    extra_vars: Dict[str, str], task: Optional[str] = None
):
    for key, val in DEFAULT_EXTRA_VARIABLES.items():
        if key not in extra_vars:
            extra_vars[key] = get_dir_from_task(val, task=task)

    return extra_vars


def start_experiment(config: str, task: str, ext_vars: Dict[str, str]):
    ext_vars = validate_extra_variables(extra_vars=ext_vars, task=task)
    try:
        result = run_experiment(
            config_path=config,
            ext_vars=ext_vars,
        )
    except Exception as e:
        trace = traceback.format_exc()
        failure_str = "Exception during training: " + str(e) + "\n" + trace
        with open(os.path.join(ext_vars["OUTPUT_PATH"], "failure"), "w") as fp:
            fp.write(failure_str)
        print(failure_str, file=sys.stderr)
    else:
        print("Training complete.")
        return result


if __name__ == "__main__":
    EXPERIMENT_NAME = "roberta-base-training-example"
    TASK = "question-answering"

    WORKING_DIR = os.getcwd()
    EXPERIMENTS_DIR = os.path.join(WORKING_DIR, "experiments")
    TASK_DIR = get_dir_from_task(os.path.join(EXPERIMENTS_DIR, "{task}"), task=TASK)
    DATASET_DIR = os.path.join(TASK_DIR, "datasets")
    EXPERIMENT_DIR = os.path.join(TASK_DIR, EXPERIMENT_NAME)
    MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model")
    CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "outputs")

    ext_vars = {
        # Used to feed the jsonnet config file with file paths
        "OUTPUT_PATH": OUTPUT_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_DIR,
    }

    CONFIG_PATH = os.path.join(
        TASK_DIR, "experiment.jsonnet"
    )  # default experiment params

    start_experiment(config=CONFIG_PATH, task=TASK, ext_vars=ext_vars)
