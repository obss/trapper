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
"""
This implementation is adapted from the token classification pipeline from the
HuggingFace's transformers library. Original code is available at:
`<https://github.com/huggingface/transformers/blob/master/src/transformers/pipelines/token_classification.py>`_.
"""
import types
from typing import Any, Dict, List, Optional, Tuple, Union

# needed for registering the data-related classes
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import src.data
from transformers import (
    ModelCard,
    PreTrainedModel,
    PreTrainedTokenizer,
    TokenClassificationPipeline,
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.pipelines import (
    SUPPORTED_TASKS,
    ArgumentHandler,
    TokenClassificationArgumentHandler,
)

from trapper.data import DataAdapter, DataCollator, DataProcessor
from trapper.pipelines import PipelineMixin


@PipelineMixin.register("example-pos-tagging", constructor="from_partial_objects")
class ExamplePosTaggingPipeline(PipelineMixin, TokenClassificationPipeline):
    """
    CONLL2003 POS tagging pipeline that extracts POS tags from a given sentence
    or a list of sentences.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        data_processor: DataProcessor,
        data_adapter: DataAdapter,
        data_collator: DataCollator,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "token-classification",
        args_parser: Optional[ArgumentHandler] = None,
        device: int = -1,
        binary_output: bool = False,
    ):
        super(ExamplePosTaggingPipeline, self).__init__(
            model=model,
            tokenizer=tokenizer,
            data_processor=data_processor,
            data_adapter=data_adapter,
            data_collator=data_collator,
            feature_extractor=feature_extractor,
            modelcard=modelcard,
            framework=framework,
            task=task,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
        )
        self._args_parser = TokenClassificationArgumentHandler()


SUPPORTED_TASKS["pos_tagging_example"] = {
    "impl": ExamplePosTaggingPipeline,
    "pt": PreTrainedModel,
}
