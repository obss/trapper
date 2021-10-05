import argparse
import os
import warnings
from typing import Dict, Optional, Tuple

import requests

from examples.util import DATASET_DIR, DEFAULT_EXTRA_VARIABLES, get_dir_from_task
from trapper.training.train import run_experiment

__arguments__ = ["config", "task", "experiment_name"]


def download_squad(
    task: str, version: str = "1.1", overwrite: bool = False
) -> Tuple[str, str]:
    """
    Downloads SQuAD dataset with given version.

    Args:
        task:
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
    result = run_experiment(
        config_path=config,
        ext_vars=ext_vars,
    )
    print("Training complete.")
    return result


def main():
    experiment_name = "roberta-base-training-example"
    task = "question-answering"
    working_dir = os.getcwd()
    experiments_dir = os.path.join(working_dir, "experiments")
    task_dir = get_dir_from_task(os.path.join(experiments_dir, "{task}"), task=task)
    experiment_dir = os.path.join(task_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = os.path.join(experiment_dir, "outputs")
    ext_vars = {
        # Used to feed the jsonnet config file with file paths
        "OUTPUT_PATH": output_dir,
        "CHECKPOINT_PATH": checkpoint_dir,
    }
    config_path = os.path.join(
        task_dir, "experiment.jsonnet"
    )  # default experiment params
    start_experiment(config=config_path, task=task, ext_vars=ext_vars)


if __name__ == "__main__":
    main()
