from typing import Dict, List

from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import TokenizerWrapper
from trapper.data.data_adapters.data_adapter import DataAdapter
from trapper.data.data_processors import IndexedInstance


@DataAdapter.register("conll2003_pos_tagging_example")
class ExampleDataAdapterForPosTagging(DataAdapter):
    # Obtained by executing `dataset["train"].features["pos_tags"].feature.names`
    _LABELS = (
        '"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT', 'EX',
        'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS',
        'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
        'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
        'WRB'
    )

    def __init__(
            self,
            tokenizer_wrapper: TokenizerWrapper,
            labels: List[str] = None
    ):
        super().__init__(tokenizer_wrapper)
        self._LABELS = labels or self._LABELS
        self._LABEL_TO_ID = {label: i for i, label in enumerate(self._LABELS)}

    def __call__(self, raw_instance: IndexedInstance) -> IndexedInstance:
        """
        Create a sequence with the following field:
        input_ids: <bos> ...tokens... <eos>
        """
        # We return an instance having the keys fields specified in
        # `trapper.models.auto_wrappers._TASK_TO_INPUT_FIELDS["token_classification"]`
        input_ids = [self._bos_token_id] + raw_instance["tokens"]
        input_ids.append(self._eos_token_id)
        labels = [IGNORED_LABEL_ID] + raw_instance["pos_tags"]
        labels.append(IGNORED_LABEL_ID)
        return {"input_ids": input_ids, "labels": labels}

    @property
    def id_to_label(self) -> Dict[int, str]:
        # Will be used by the pipeline object for inference
        return {i: label for i, label in enumerate(self._LABELS)}
