import os
from typing import Dict, Optional

from trapper.training.train import run_experiment

from examples.question_answering.util import (
    DEFAULT_EXTRA_VARIABLES,
    get_dir_from_task,
)


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
