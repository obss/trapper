import logging
from typing import Any, Dict, List, Optional

from trapper.common.constants import SpanTuple
from trapper.data.data_processors import DataProcessor
from trapper.data.data_processors.data_processor import (
    ImproperDataInstanceError,
    IndexedInstance,
)

logger = logging.getLogger(__file__)


@DataProcessor.register("conll2003_pos_tagging_example")
class ExampleConll2003PosTaggingDataProcessor(DataProcessor):
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 2  # <bos> tokens <eos>

    def process(self, instance_dict: Dict[str, Any]) -> Optional[IndexedInstance]:
        id_ = instance_dict["id"]
        tokens = instance_dict["tokens"]
        pos_tags = instance_dict["pos_tags"]
        return self.text_to_instance(
            id_=id_,
            tokens=tokens,
            pos_tags=pos_tags,
        )

    def text_to_instance(
            self,
            id_: str,
            tokens: List[str],
            pos_tags: Optional[List[int]] = None,
    ) -> IndexedInstance:
        expanded_tokens = []
        expanded_pos_tags = []
        for token, pos_tag in zip(tokens, pos_tags):
            expanded_token = self.tokenizer.tokenize(token)
            expanded_pos_tag = [pos_tag] * len(expanded_token)
            expanded_tokens.append(expanded_token)
            expanded_pos_tags.append(expanded_pos_tag)

        for seq in (expanded_tokens, expanded_pos_tags):
            self._chop_excess_tokens(seq, len(seq))

        return {
            "tokens": self.tokenizer.convert_tokens_to_ids(expanded_tokens),
            "pos_tags": expanded_pos_tags
        }

    @staticmethod
    def filtered_instance() -> IndexedInstance:
        return {
            "context": [],
            "answer_positions": [],
            "answer_tokens": [],
            "context_index": -1,
            "__discard_sample": True,
        }
