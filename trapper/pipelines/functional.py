from pathlib import Path
from typing import Union, Dict, Any, Optional

from trapper.common.params import Params
from trapper.pipelines.pipeline import Pipeline, PIPELINE_CONFIG_ARGS


def _read_pipeline_params(
    config_path: str,
    params_overrides: Union[str, Dict[str, Any]],
) -> Params:
    if not (config_path.endswith(".json") or config_path.endswith(".jsonnet")):
        raise ValueError(
            "Illegal file format. Please provide a json or jsonnet file!"
        )
    _validate_params_overrides(params_overrides)
    params = Params.from_file(
        params_file=config_path,
        params_overrides=params_overrides,
    )
    data_components = params.get("dataset_loader").params
    params.update(data_components)
    params = {k: v for k, v in params.items() if k in PIPELINE_CONFIG_ARGS}
    return Params(params)


def _validate_checkpoint_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Input path must be an existing directory")


def _validate_params_overrides(params_overrides: Union[str, Dict[str, Any]]) -> None:
    if params_overrides is None:
        return
    elif isinstance(params_overrides, dict):
        if "type" in params_overrides or "pretrained_model_name_or_path" in params_overrides:
            raise ValueError("'type' and 'pretrained_model_name_or_path are not allowed "
                             "to be used in 'params_overrides'.")


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
    params.update({"type": pipeline_type, "pretrained_model_name_or_path": checkpoint_path})
    return create_pipeline_from_params(params)
