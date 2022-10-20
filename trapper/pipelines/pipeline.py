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
from abc import abstractmethod, ABC
from typing import Any, Dict, Optional, final, Tuple

from transformers import ModelCard
from transformers import Pipeline as _Pipeline
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.pipelines import ArgumentHandler as _ArgumentHandler, pipeline
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

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
]


class ArgumentHandler(_ArgumentHandler, Registrable):
    """
    Registered ArgumentHandler class for pipeline class/subclasses.
    """


@append_parent_docstr(parent_id=0)
class Pipeline(_Pipeline, Registrable):

    default_implementation = "default"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data_processor: DataProcessor,
        data_adapter: DataAdapter,
        data_collator: DataCollator,
        args_parser: Optional[ArgumentHandler] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        device: int = -1,
        binary_output: bool = False,
    ):
        super(Pipeline, self).__init__(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            modelcard=modelcard,
            framework=framework,
            task=task,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
        )
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
        data_collator: Lazy[DataCollator],
        args_parser: Optional[ArgumentHandler] = None,
        model_max_sequence_length: Optional[int] = None,
        label_mapper: Optional[LabelMapper] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        device: int = -1,
        binary_output: bool = False,
    ) -> "Pipeline":

        #  To find the registrable components from the user-defined packages
        import_plugins()

        model_wrapper_ = model_wrapper.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        model_forward_params = model_wrapper_.forward_params

        tokenizer_wrapper_ = tokenizer_wrapper.construct(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

        data_processor_ = data_processor.construct(
            tokenizer_wrapper=tokenizer_wrapper_,
            label_mapper=label_mapper,
            model_max_sequence_length=model_max_sequence_length,
        )

        data_collator_ = data_collator.construct(
            tokenizer_wrapper=tokenizer_wrapper_,
            model_forward_params=model_forward_params,
        )

        data_adapter_ = data_adapter.construct(
            tokenizer_wrapper=tokenizer_wrapper_, label_mapper=label_mapper
        )

        return cls(
            model=model_wrapper_.model,
            tokenizer=tokenizer_wrapper_.tokenizer,
            data_processor=data_processor_,
            data_adapter=data_adapter_,
            data_collator=data_collator_,
            args_parser=args_parser,
            feature_extractor=feature_extractor,
            modelcard=modelcard,
            framework=framework,
            task=task,
            device=device,
            binary_output=binary_output,
        )

    @final
    def __call__(self, *args, **kwargs):
        """
        This method should not be overridden.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        # Convert inputs to features
        examples = self._args_parser(*args, **kwargs)
        if len(examples) == 1:
            return super().__call__(examples[0], **kwargs)
        return super().__call__(examples, **kwargs)

    def preprocess(self, example: Any, **preprocess_kwargs) -> Dict[str, GenericTensor]:
        """
        Preprocessing utilizing data components. This method can be overridden in child
        classes.

        Args:
            example: A dataset, sample of instances or a single instance to be processed.
            **preprocess_kwargs: Additional keyword arguments for preprocess.

        Returns:
            A dictionary making up the model inputs.
        """
        indexed_instance = self.data_processor.text_to_instance(**example)
        indexed_instance = self.data_adapter(indexed_instance)
        return {"indexed_instance": indexed_instance, "example": example}

    # @abstractmethod
    # def _sanitize_parameters(self, **pipeline_parameters) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    #     pass
    #
    # @abstractmethod
    # def _forward(
    #     self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict
    # ) -> ModelOutput:
    #     pass
    #
    # @abstractmethod
    # def postprocess(
    #     self, model_outputs: ModelOutput, **postprocess_parameters: Dict
    # ) -> Any:
    #     pass


ArgumentHandler.register("default")(ArgumentHandler)

Pipeline.register("default", constructor="from_partial_objects")(Pipeline)
