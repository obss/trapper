import os
from typing import Dict

from trapper.training.train import run_experiment


def start_experiment(config: str, ext_vars: Dict[str, str]):
    result = run_experiment(
        config_path=config,
        ext_vars=ext_vars,
    )
    print("Training complete.")
    return result


def main():
    experiment_name = "roberta-base-training-example"
    working_dir = os.path.dirname(os.getcwd())
    experiment_dir = os.path.join(working_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = os.path.join(experiment_dir, "outputs")
    ext_vars = {
        # Used to feed the jsonnet config file with file paths
        "OUTPUT_PATH": output_dir,
        "CHECKPOINT_PATH": checkpoint_dir,
    }
    config_path = os.path.join(
        working_dir, "experiment.jsonnet"
    )  # default experiment params
    start_experiment(config=config_path, ext_vars=ext_vars)


if __name__ == "__main__":
    main()
