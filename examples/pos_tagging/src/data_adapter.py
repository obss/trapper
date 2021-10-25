from examples.pos_tagging.src.label_mapper import ExampleLabelMapperForPosTagging
from trapper.common.constants import IGNORED_LABEL_ID
from trapper.data import TokenizerWrapper
from trapper.data.data_adapters.data_adapter import DataAdapter
from trapper.data.data_processors import IndexedInstance


@DataAdapter.register("conll2003_pos_tagging_example")
class ExampleDataAdapterForPosTagging(DataAdapter):
    """
    This class takes the processed instance dict from the data processor and
    creates a new dict that has the "input_ids" and "labels" keys required by the
    models. It also takes care of the special BOS and EOS tokens while constructing
    these fields.
    """

    def __init__(
            self,
            tokenizer_wrapper: TokenizerWrapper,
    ):
        label_mapper = ExampleLabelMapperForPosTagging.from_labels()
        super().__init__(tokenizer_wrapper, label_mapper)

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
