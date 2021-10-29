"""
Answer extraction pipeline that can be used to extract answer in the span
format from a context string (e.g. sentence or paragraph) in question
generation task.

This implementation wraps the TokenClassificationPipeline from the
HuggingFace's transformers library.
"""
from typing import List, Optional, Union

import numpy as np
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
from trapper.data import LabelMapper
from trapper.pipelines.pipeline import create_pipeline_from_checkpoint


class ExamplePosTaggingPipeline(TokenClassificationPipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            label_mapper: LabelMapper,
            modelcard: Optional[ModelCard] = None,
            framework: Optional[str] = None,
            args_parser: ArgumentHandler = TokenClassificationArgumentHandler(),
            device: int = -1,
            binary_output: bool = False,
            task: str = "",
            grouped_entities: bool = False,
            ignore_subwords: bool = False,
            **kwargs,  # For the ignored arguments
    ):
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

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `grouped_entities` is set to True.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        inputs, offset_mappings = self._args_parser(inputs, **kwargs)
        _inputs = []
        for inp in inputs:
            words = [item[0] for item in
                     self.whitespace_tokenizer.pre_tokenize_str(inp)]
            _inputs.append(words)

        answers = []

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    truncation=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                    is_split_into_words=True
                )
                if self.tokenizer.is_fast:
                    offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = \
                    tokens.pop("special_tokens_mask").cpu().numpy()[0]

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens.data)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities = self.model(**tokens)[0][0].cpu().numpy()
                        input_ids = tokens["input_ids"].cpu().numpy()[0]

            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            # Filter to labels not in `self.ignore_labels`
            # Filter special_tokens
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if (self.model.config.id2label[
                        label_idx] not in self.ignore_labels) and not
                   special_tokens_mask[idx]
            ]

            for idx, label_idx in filtered_labels_idx:
                if offset_mapping is not None:
                    start_ind, end_ind = offset_mapping[idx]
                    word_ref = sentence[start_ind:end_ind]
                    word = \
                        self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[
                            0]
                    is_subword = len(word_ref) != len(word)

                    if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                        word = word_ref
                        is_subword = False
                else:
                    word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

                    start_ind = None
                    end_ind = None

                entity = {
                    "word": word,
                    "score": score[idx][label_idx].item(),
                    "entity": self.model.config.id2label[label_idx],
                    "index": idx,
                    "start": start_ind,
                    "end": end_ind,
                }

                if self.grouped_entities and self.ignore_subwords:
                    entity["is_subword"] = is_subword

                entities += [entity]

            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                    (
                            entity["entity"].split("-")[-1] ==
                            entity_group_disagg[-1]["entity"].split("-")[-1]
                            and entity["entity"].split("-")[0] != "B"
                    )
                    and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[
                        -1]
                    entity[
                        "score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups


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
    print(output)
    return output


if __name__ == '__main__':
    main()
