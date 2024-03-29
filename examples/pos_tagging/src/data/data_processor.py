import logging
from typing import Any, Dict, List, Optional

from trapper.data.data_processors import DataProcessor
from trapper.data.data_processors.data_processor import IndexedInstance

logger = logging.getLogger(__file__)


@DataProcessor.register("conll2003_pos_tagging_example")
class ExampleConll2003PosTaggingDataProcessor(DataProcessor):
    """
    This class extracts the "tokens", "pos_tags" and "id" fields from from an input
    data instance. It tokenizes the `tokens` field since it actually consists of
    words which may need further tokenization. Then, it generates the corresponding
    token ids and store them. Finally, the`pos_tags` are stored directly without
    any processing since this field consists of integer labels ids instead of
    categorical labels.
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
        tokens: List[str],
        id_: str = 0,
        pos_tags: Optional[List[int]] = None,
    ) -> IndexedInstance:
        expanded_tokens = []
        expanded_token_counts = []
        for token in tokens:
            expanded_token = self.tokenizer.tokenize(token)
            expanded_tokens.extend(expanded_token)
            expanded_token_counts.append(len(expanded_token))

        instance = {"id": id_}

        if pos_tags is not None:
            expanded_pos_tags = []
            for expanded_len, pos_tag in zip(expanded_token_counts, pos_tags):
                expanded_pos_tag = [pos_tag] * expanded_len
                expanded_pos_tags.extend(expanded_pos_tag)
            self._chop_excess_tokens(expanded_pos_tags, len(expanded_pos_tags))
            instance["pos_tags"] = expanded_pos_tags

        self._chop_excess_tokens(expanded_tokens, len(expanded_tokens))
        instance["tokens"] = self.tokenizer.convert_tokens_to_ids(expanded_tokens)

        return instance
