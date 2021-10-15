import logging
import os
import sys
from os import PathLike
from typing import Any, Dict, List, Union

import transformers
from transformers.trainer_utils import is_main_process

from trapper.common import Params
from trapper.training import TransformerTrainer

logger = logging.getLogger(__name__)

ARCHIVED_CONFIG_NAME = "experiment_config.json"


def run_experiment(
    config_path: Union[str, PathLike],
    params_overrides: Union[str, Dict[str, Any]] = "",
    ext_vars: dict = None,
) -> Dict[str, float]:
    """
    The first function called by the `trapper.training.train` module after
    `trapper` is invoked with the `run` command form the CLI or through directly
    running the module as a script.
    Reads the experiment details from the config file and initiates a training
    and/or evaluation run.

    Args:
        config_path (): The path to the config file specifying the experiment.
        params_overrides (): The external parameters specified to override the
            experiment parameters
        ext_vars (): The values made available while reading the config file
            thanks to the Jsonnet's external variable injection mechanism.

    Returns:
        Experiment's results e.g. the metric values in a dict
    """
    params = _read_experiment_params(str(config_path), params_overrides, ext_vars)
    return _run_experiment_from_params(params)


def _read_experiment_params(
    config_path: str,
    params_overrides: Union[str, Dict[str, Any]] = "",
    ext_vars: dict = None,
) -> Params:
    if not (config_path.endswith(".json") or config_path.endswith(".jsonnet")):
        raise ValueError(
            "Illegal file format. Please provide a json or jsonnet file!"
        )
    params = Params.from_file(
        params_file=config_path,
        params_overrides=params_overrides,
        ext_vars=ext_vars,
    )
    return params


def _run_experiment_from_params(params: Params) -> Dict[str, float]:
    serialization_dirs = _create_serialization_dirs(params)
    _save_experiment_config(params, serialization_dirs)
    trainer = TransformerTrainer.from_params(params)
    return run_experiment_using_trainer(trainer)


def _create_serialization_dirs(params: Params) -> List[str]:
    serialization_dirs = [
        params.params["args"].get(key, None) for key in ("output_dir", "result_dir")
    ]
    if None in serialization_dirs:
        raise ValueError(
            'You must supply an "output_dir" and a "result_dir" for the experiment.'
        )

    for dir_ in serialization_dirs:
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        elif len(os.listdir(dir_)) > 0:
            raise ValueError(
                f"Serialization directory: `{dir_}` not emtpy. Provide an "
                f"empty or non-existent directory."
            )

    return serialization_dirs


def _save_experiment_config(params: Params, serialization_dirs: List[str]):
    for dir_ in serialization_dirs:
        params.to_file(os.path.join(dir_, ARCHIVED_CONFIG_NAME))


def run_experiment_using_trainer(trainer: TransformerTrainer) -> Dict[str, float]:
    """
    The function is used to start an experiment i.e. a training and/or evaluation
    run using the values set in the trainer object.

    Args:
        trainer (): The object holding all experiment details including the
            model, data-related classes, callbacks etc.

    Returns:
        Experiment's results e.g. the metric values in a dict
    """
    training_args = trainer.args

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    result_dir = training_args.result_dir
    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model(result_dir)  # Saves the tokenizer too for easy upload

        train_results_file = os.path.join(result_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(train_results_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only
            # the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(result_dir, "trainer_state.json")
            )

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        eval_results_file = os.path.join(result_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(eval_results_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def main():
    """
    Invokes a training and/or evaluation experiment specified in the input config
    file when the `trapper.training.train` module is run as a script.

    Returns:
        Experiment's results e.g. the metric values in a dict
    """
    if not len(sys.argv) == 2:
        raise ValueError(
            "`train` script accepts a single argument i.e. the config file in json"
            " or jsonnet format!"
        )
    config_path = sys.argv[1]
    return run_experiment(config_path)


if __name__ == "__main__":
    main()
