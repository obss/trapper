# Copyright 2021 Open Business Software Solutions, the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from transformers import Pipeline as _Pipeline

from trapper.common import Lazy, Registrable
from trapper.common.plugins import import_plugins
from trapper.common.utils import append_parent_docstr
from trapper.data import (
    DataAdapter,
    DataCollator,
    DataProcessor,
    LabelMapper,
    TokenizerWrapper,
)
from trapper.models import ModelWrapper

PIPELINE_CONFIG_ARGS = [
    "pretrained_model_name_or_path",
    "model_wrapper",
    "tokenizer_wrapper",
    "data_processor",
    "data_adapter",
    "data_collator",
    "args_parser",
    "model_max_sequence_length",
    "label_mapper",
    "feature_extractor",
    "modelcard",
    "framework",
    "task",
    "device",
    "binary_output",
    "use_auth_token",
]


@append_parent_docstr(parent_id=0)
class PipelineMixin(_Pipeline, Registrable):
    """
    A Mixin class for constructing pipelines that utilize data components of trapper.
    This class' precedence in multiple inheritance should be higher, i.e. inheritance
    order of PipelineMixin should be low.

    Note:
        In theory and practice this class can be used as a base class to create a
        custom concrete class; however, this class is designed as a mixin to be used
        with transformers' pipeline classes, and should never be used solely as it
        is not a concrete class.

        Although not recommended, it can be used like a base class that extends
        transformers Pipeline class. In this case, this class must implement
        the abstract and required methods.

    Examples:
        from transformers.pipelines import QuestionAnsweringPipeline

        class CustomQAPipeline(PipelineMixin, QuestionAnsweringPipeline):
            ...
    """

    default_implementation = "default"

    def __init__(
        self,
        data_processor: DataProcessor,
        data_adapter: DataAdapter,
        data_collator: Optional[DataCollator] = None,
        **kwargs
    ):
        super(PipelineMixin, self).__init__(**kwargs)
        self._data_processor = data_processor
        self._data_adapter = data_adapter
        self._data_collator = data_collator

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def data_adapter(self):
        return self._data_adapter

    @property
    def data_collator(self):
        return self._data_collator

    @classmethod
    def from_partial_objects(
        cls,
        pretrained_model_name_or_path: str,
        model_wrapper: Lazy[ModelWrapper],
        tokenizer_wrapper: Lazy[TokenizerWrapper],
        data_processor: Lazy[DataProcessor],
        data_adapter: Lazy[DataAdapter],
        data_collator: Optional[Lazy[DataCollator]] = None,
        label_mapper: Optional[Lazy[LabelMapper]] = None,
        model_max_sequence_length: Optional[int] = None,
        framework: Optional[str] = "pt",
        task: str = "",
        device: int = -1,
        binary_output: bool = False,
        use_auth_token: bool = None,
        **kwargs
    ) -> "PipelineMixin":

        #  To find the registrable components from the user-defined packages
        import_plugins()

        model_wrapper_ = model_wrapper.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_auth_token=use_auth_token,
        )
        model_forward_params = model_wrapper_.forward_params

        if label_mapper:
            label_mapper_ = label_mapper.construct()
        else:
            label_mapper_ = None

        tokenizer_wrapper_ = tokenizer_wrapper.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_auth_token=use_auth_token,
        )

        data_processor_ = data_processor.construct(
            tokenizer_wrapper=tokenizer_wrapper_,
            label_mapper=label_mapper_,
            model_max_sequence_length=model_max_sequence_length,
        )

        if data_collator:
            data_collator_ = data_collator.construct(
                tokenizer_wrapper=tokenizer_wrapper_,
                model_forward_params=model_forward_params,
            )
        else:
            data_collator_ = None

        data_adapter_ = data_adapter.construct(
            tokenizer_wrapper=tokenizer_wrapper_, label_mapper=label_mapper
        )

        return cls(
            model=model_wrapper_.model,
            tokenizer=tokenizer_wrapper_.tokenizer,
            data_processor=data_processor_,
            data_adapter=data_adapter_,
            data_collator=data_collator_,
            framework=framework,
            task=task,
            device=device,
            binary_output=binary_output,
            **kwargs
        )


PipelineMixin.register("default", constructor="from_partial_objects")(PipelineMixin)
