from pathlib import Path
from typing import Union

from transformers import AutoConfig, Pipeline
from transformers.pipelines import pipeline

from trapper.common.params import Params
from trapper.data import DataAdapter, DataProcessor, TransformerTokenizer
from trapper.data.data_collator import DataCollator
from trapper.models import TransformerModel


def create_pipeline_from_checkpoint(
    checkpoint_path: Union[str, Path],
    experiment_config_path: Union[str, Path],
    task: str,
    **kwargs
) -> Pipeline:
    _validate_checkpoint_dir(checkpoint_path)
    params = Params.from_file(params_file=experiment_config_path).params
    pipeline_ = _create_pipeline(checkpoint_path, params, task, **kwargs)
    return pipeline_


def _create_pipeline(checkpoint_path, params, task: str, **kwargs):
    model = _create_model(checkpoint_path, params)
    tokenizer = _create_tokenizer(checkpoint_path, params)
    data_processor = _create_data_processor(params, tokenizer)
    data_adapter = _create_data_adapter(params, tokenizer)
    data_collator = _create_data_collator(model, tokenizer)
    config = AutoConfig.from_pretrained(checkpoint_path)
    pipeline_ = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        config=config,
        framework="pt",
        data_processor=data_processor,
        data_adapter=data_adapter,
        data_collator=data_collator,
        **kwargs
    )
    return pipeline_


def _create_data_collator(model, tokenizer):
    return DataCollator(
        tokenizer=tokenizer, model_forward_params=model.forward_params
    )


def _validate_checkpoint_dir(path: Union[str, Path]):
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Input path must be an existing directory")


def _create_model(checkpoint_path, params):
    return TransformerModel.from_params(
        Params(
            {
                "type": params["model"]["type"],
                "pretrained_model_name_or_path": checkpoint_path,
            }
        )
    )


def _create_tokenizer(checkpoint_path, params):
    return TransformerTokenizer.from_params(
        Params(
            {
                "type": params["tokenizer"]["type"],
                "pretrained_model_name_or_path": checkpoint_path,
            }
        )
    )


def _create_data_processor(params, tokenizer):
    return DataProcessor.by_name(
        params["dataset_loader"]["data_processor"]["type"]
    )(tokenizer)


def _create_data_adapter(params, tokenizer):
    return DataAdapter.by_name(params["dataset_loader"]["data_adapter"]["type"])(
        tokenizer
    )
