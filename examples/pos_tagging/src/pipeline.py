"""
Answer extraction pipeline that can be used to extract answer in the span
format from a context string (e.g. sentence or paragraph) in question
generation task.

This implementation wraps the TokenClassificationPipeline from the
HuggingFace's transformers library.
"""
from typing import Optional

import torch
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    ModelCard,
    PreTrainedModel,
    PreTrainedTokenizer,
    TokenClassificationPipeline,
)
from transformers.pipelines import (
    SUPPORTED_TASKS,
    ArgumentHandler,
    TokenClassificationArgumentHandler,
)

# needed for registering the data-related classes
# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import examples.pos_tagging.src
from trapper import PROJECT_ROOT
from trapper.common.utils import append_parent_docstr
from trapper.data import DataAdapter, DataCollator, DataProcessor, LabelMapper
from trapper.pipelines.pipeline import create_pipeline_from_checkpoint


@append_parent_docstr
class ExamplePosTaggingPipeline(TokenClassificationPipeline):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            label_mapper: LabelMapper,
            data_processor: DataProcessor,
            data_adapter: DataAdapter,
            data_collator: DataCollator,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = "pt",
            args_parser: ArgumentHandler = None,
            device: int = -1,
            binary_output: bool = False,
            task: str = "",
            grouped_entities: bool = False,
            ignore_subwords: bool = False,
            **kwargs,  # For the ignored arguments
    ):
        args_parser = args_parser or TokenClassificationArgumentHandler()
        self.label_mapper = label_mapper
        self.data_processor = data_processor
        self.data_adapter = data_adapter
        self.data_collator = data_collator
        self.whitespace_tokenizer = Whitespace()
        model.config.id2label = label_mapper.id_to_label_map
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=binary_output,
            task=task,
            grouped_entities=grouped_entities,
            ignore_subwords=ignore_subwords,
        )


SUPPORTED_TASKS["pos_tagging_example"] = {
    "impl": ExamplePosTaggingPipeline,
    "pt": PreTrainedModel,
}


def main():
    pos_tagging_project_root = PROJECT_ROOT / "examples/pos_tagging"
    checkpoints_dir = (pos_tagging_project_root /
                       "outputs/roberta/outputs/checkpoints")
    pipeline = create_pipeline_from_checkpoint(
        checkpoint_path=checkpoints_dir / "checkpoint-2628",
        experiment_config_path=checkpoints_dir / "experiment_config.json",
        task="pos_tagging_example",
        grouped_entities=False,
        ignore_subwords=False,
    )
    output = pipeline("I love Berlin.")
    return output


if __name__ == '__main__':
    main()
