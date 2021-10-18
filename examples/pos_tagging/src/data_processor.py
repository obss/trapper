import logging
from typing import Any, Dict, List, Optional

from trapper.data.data_processors import DataProcessor
from trapper.data.data_processors.data_processor import IndexedInstance

logger = logging.getLogger(__file__)


@DataProcessor.register("conll2003_pos_tagging_example")
class ExampleConll2003PosTaggingDataProcessor(DataProcessor):
    """
    This class extracts the "tokens", "pos_tags" and "id" fields from the
    a given data instance. The tokens are tokenized and the token ids are stored
    whereas the pos tags are used as they are since they are already in `int`
    format.
    """
    NUM_EXTRA_SPECIAL_TOKENS_IN_SEQUENCE = 2  # <bos> tokens <eos>

    def process(self, instance_dict: Dict[str, Any]) -> Optional[IndexedInstance]:
        return self.text_to_instance(
            id_=instance_dict["id"],
            tokens=instance_dict["tokens"],
            pos_tags=instance_dict["pos_tags"],
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
            expanded_tokens.extend(expanded_token)
            expanded_pos_tags.extend(expanded_pos_tag)

        for seq in (expanded_tokens, expanded_pos_tags):
            self._chop_excess_tokens(seq, len(seq))

        return {
            "tokens": self.tokenizer.convert_tokens_to_ids(expanded_tokens),
            "pos_tags": expanded_pos_tags
        }
