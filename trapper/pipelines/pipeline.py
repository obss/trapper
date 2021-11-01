from pathlib import Path
from typing import Optional, Union

from transformers import AutoConfig, Pipeline
from transformers.pipelines import pipeline

from trapper.common.params import Params
from trapper.data import DataAdapter, DataProcessor, TokenizerWrapper
from trapper.data.data_collator import DataCollator
from trapper.data.label_mapper import LabelMapper
from trapper.models import ModelWrapper


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


def _create_pipeline(checkpoint_path, params: Params, task: str, **kwargs):
    model_wrapper = _create_model_wrapper(checkpoint_path, params)
    tokenizer_wrapper = _create_tokenizer(checkpoint_path, params)
    label_mapper = _create_label_mapper(params)
    data_processor = _create_data_processor(params, tokenizer_wrapper, label_mapper)
    data_adapter = _create_data_adapter(params, tokenizer_wrapper, label_mapper)
    data_collator = _create_data_collator(model_wrapper, tokenizer_wrapper)
    config = AutoConfig.from_pretrained(checkpoint_path)
    pipeline_ = pipeline(
        task=task,
        model=model_wrapper.model,
        tokenizer=tokenizer_wrapper.tokenizer,
        config=config,
        framework="pt",
        label_mapper=label_mapper,
        data_processor=data_processor,
        data_adapter=data_adapter,
        data_collator=data_collator,
        **kwargs
    )
    return pipeline_


def _create_data_collator(
    model_wrapper: ModelWrapper, tokenizer_wrapper: TokenizerWrapper
) -> DataCollator:
    return DataCollator(
        tokenizer_wrapper=tokenizer_wrapper,
        model_forward_params=model_wrapper.forward_params,
    )


def _validate_checkpoint_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if not path.is_dir():
        raise ValueError("Input path must be an existing directory")


def _create_model_wrapper(checkpoint_path, params: Params) -> ModelWrapper:
    return ModelWrapper.from_params(
        Params(
            {
                "type": params["model_wrapper"]["type"],
                "pretrained_model_name_or_path": checkpoint_path,
            }
        )
    )


def _create_tokenizer(checkpoint_path, params: Params) -> TokenizerWrapper:
    return TokenizerWrapper.from_params(
        Params(
            {
                "type": params["tokenizer_wrapper"]["type"],
                "pretrained_model_name_or_path": checkpoint_path,
            }
        )
    )


def _create_label_mapper(params: Params) -> Optional[LabelMapper]:
    label_mapper_params = params.get("label_mapper")
    if label_mapper_params is None:
        return None
    constructor = LabelMapper.by_name(label_mapper_params["type"])
    del label_mapper_params["type"]
    return constructor(**label_mapper_params)


def _create_data_processor(
    params: Params, tokenizer_wrapper: TokenizerWrapper, label_mapper: LabelMapper
) -> DataProcessor:
    data_processor_params = params["dataset_loader"]["data_processor"]
    constructor = DataProcessor.by_name(data_processor_params["type"])
    model_max_sequence_length = data_processor_params.get(
        "model_max_sequence_length"
    )
    return constructor(
        tokenizer_wrapper=tokenizer_wrapper,
        model_max_sequence_length=model_max_sequence_length,
        label_mapper=label_mapper,
    )


def _create_data_adapter(
    params: Params, tokenizer_wrapper: TokenizerWrapper, label_mapper: LabelMapper
) -> DataAdapter:
    constructor = DataAdapter.by_name(
        params["dataset_loader"]["data_adapter"]["type"]
    )
    return constructor(
        tokenizer_wrapper=tokenizer_wrapper, label_mapper=label_mapper
    )
