from pathlib import Path
from typing import Any, Dict, Optional, Union

from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from trapper.common.params import Params
from trapper.pipelines.pipeline import PIPELINE_CONFIG_ARGS, PipelineMixin


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
    return params


def _validate_checkpoint_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Input path must be an existing directory")


def _validate_params_overrides(
    params_overrides: Union[str, Dict[str, Any]]
) -> None:
    if params_overrides is None:
        return
    elif isinstance(params_overrides, dict):
        if (
            "type" in params_overrides
            or "pretrained_model_name_or_path" in params_overrides
        ):
            raise ValueError(
                "'type' and 'pretrained_model_name_or_path are not allowed "
                "to be used in 'params_overrides'."
            )


def create_pipeline_from_params(
    params,
    pipeline_type: Optional[str] = "default",
    pretrained_model_name_or_path: Optional[str] = None,
) -> PipelineMixin:
    data_components = params.get("dataset_loader").params
    params.update(data_components)
    params = Params({k: v for k, v in params.items() if k in PIPELINE_CONFIG_ARGS})
    params.update(
        {
            "type": pipeline_type,
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
        }
    )
    return PipelineMixin.from_params(params)


def repo_exists(repo_id: str) -> bool:
    hf_api = HfApi()
    try:
        hf_api.repo_info(repo_id)
    except RepositoryNotFoundError:
        return False
    return True


def _sanitize_checkpoint(
    checkpoint_path: Union[str, Path],
    experiment_config_path: Union[str, Path, None],
    use_auth_token: Union[bool, str, None],
) -> str:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        if experiment_config_path is None:
            raise ValueError(
                "`experiment_config_path` cannot be None if `checkpoint_path` is a local_directory."
            )
    elif repo_exists(checkpoint_path.as_posix()):
        if experiment_config_path is None:
            try:
                experiment_config_path = hf_hub_download(
                    checkpoint_path.as_posix(),
                    "experiment_config.json",
                    use_auth_token=use_auth_token,
                )
            except EntryNotFoundError:
                raise ValueError(
                    "If a model is given in HF-hub, `experiment_config.json` must be included in "
                    "the model hub repository."
                )
    else:
        raise ValueError(
            "Input path must be an existing directory or an existing "
            "repository at huggingface model hub."
        )
    return experiment_config_path


def create_pipeline_from_checkpoint(
    checkpoint_path: Union[str, Path],
    experiment_config_path: Union[str, Path] = None,
    params_overrides: Union[str, Dict[str, Any]] = None,
    pipeline_type: Optional[str] = "default",
    use_auth_token: Union[str, bool, None] = None,
) -> PipelineMixin:
    experiment_config_path = _sanitize_checkpoint(
        checkpoint_path, experiment_config_path, use_auth_token=use_auth_token
    )
    params = _read_pipeline_params(experiment_config_path, params_overrides)
    return create_pipeline_from_params(
        params,
        pipeline_type=pipeline_type,
        pretrained_model_name_or_path=checkpoint_path,
    )
