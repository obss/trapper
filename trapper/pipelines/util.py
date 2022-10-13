from pathlib import Path
from typing import Union, Dict, Any, Optional

from trapper.common.params import Params
from trapper.pipelines.pipeline import Pipeline


def _read_pipeline_params(
    config_path: str,
    params_overrides: Union[str, Dict[str, Any]],
) -> Params:
    if not (config_path.endswith(".json") or config_path.endswith(".jsonnet")):
        raise ValueError(
            "Illegal file format. Please provide a json or jsonnet file!"
        )
    params = Params.from_file(
        params_file=config_path,
        params_overrides=params_overrides,
    )
    data_components = params.get("dataset_loader").params
    data_components = {k: v for k, v in data_components.items() if k.startswith("data_")}
    params.update(data_components)

    return params


def _validate_checkpoint_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Input path must be an existing directory")


def create_pipeline_from_params(params) -> Pipeline:
    return Pipeline.from_params(params)


def create_pipeline_from_checkpoint(
    checkpoint_path: Union[str, Path],
    experiment_config_path: Union[str, Path],
    params_overrides: Union[str, Dict[str, Any]] = None,
    pipeline_type: Optional[str] = "default",
) -> Pipeline:
    _validate_checkpoint_dir(checkpoint_path)
    params = _read_pipeline_params(experiment_config_path, params_overrides)
    params.update({"type": pipeline_type})
    return create_pipeline_from_params(params)


if __name__ == "__main__":
    import os
    from trapper import PROJECT_ROOT
    from trapper.training.train import run_experiment

    experiment_dir = "/home/devrim/lab/abc"
    config_path = PROJECT_ROOT / "examples/question_answering/experiment.jsonnet"

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = os.path.join(experiment_dir, "outputs")

    ext_vars = {
        # Used to feed the jsonnet config file with file paths
        "OUTPUT_PATH"    : output_dir,
        "CHECKPOINT_PATH": checkpoint_dir
    }

    result = run_experiment(
            config_path=str(config_path),
            ext_vars=ext_vars,
    )

    PRETRAINED_MODEL_PATH = output_dir
    EXPERIMENT_CONFIG = os.path.join(PRETRAINED_MODEL_PATH, "experiment_config.json")

    qa_pipeline = create_pipeline_from_checkpoint(
            checkpoint_path=PRETRAINED_MODEL_PATH,
            experiment_config_path=EXPERIMENT_CONFIG,
            pipeline_type="squad-question-answering"
    )
