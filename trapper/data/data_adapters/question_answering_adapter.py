from typing import List, Tuple

from trapper.data.data_adapters.data_adapter import DataAdapter
from trapper.data.data_processors import IndexedInstance
from trapper.data.tokenizers.tokenizer import TransformerTokenizer


@DataAdapter.register("question-answering")
class DataAdapterForQuestionAnswering(DataAdapter):
    """
    `DataAdapterForQuestionAnswering` can be used in SQuAD style question
    answering tasks that involves a context, question and answer.
    Args:
        tokenizer ():
        model_forward_params ():
    """

    CONTEXT_TOKEN_TYPE_ID = 0
    QUESTION_TOKEN_TYPE_ID = 1

    def __init__(
            self,
            tokenizer: TransformerTokenizer,
            model_forward_params: Tuple[str, ...],
    ):
        super().__init__(tokenizer, model_forward_params)
        self._eos_token_id = self._tokenizer.eos_token_id
        self._bos_token_id = self._tokenizer.bos_token_id

    def __call__(self, raw_instance: IndexedInstance) -> IndexedInstance:
        """
        Create a sequence with the following fields:
        input_ids: <bos> ...context_toks... <eos> ...question_toks... <eos>
        token_type_ids: 0 for context tokens, 1 for question tokens.
        """
        instance = self._build_context(raw_instance)
        self._append_separator_token(instance)
        self._append_question_tokens(instance=instance, raw_instance=raw_instance)
        self._append_ending_token(instance)
        return instance

    def _build_context(self, raw_instance: IndexedInstance) -> IndexedInstance:
        context_tokens = raw_instance["context"]
        input_ids = [self._bos_token_id] + context_tokens
        token_type_ids = self._context_token_type_ids(context_tokens)
        instance = {"input_ids": input_ids, "token_type_ids": token_type_ids}
        self._handle_answer_span(instance, raw_instance)
        return instance

    def _append_separator_token(self, instance: IndexedInstance):
        self._extend_token_ids(
            instance=instance,
            token_type_id=self.CONTEXT_TOKEN_TYPE_ID,
            input_ids=[self._eos_token_id],
        )

    def _append_question_tokens(
            self, instance: IndexedInstance, raw_instance: IndexedInstance
    ):
        self._extend_token_ids(
            instance=instance,
            token_type_id=self.QUESTION_TOKEN_TYPE_ID,
            input_ids=raw_instance["question"],
        )

    def _append_ending_token(self, instance: IndexedInstance):
        self._extend_token_ids(
            instance=instance,
            token_type_id=self.QUESTION_TOKEN_TYPE_ID,
            input_ids=[self._eos_token_id],
        )

    def _context_token_type_ids(self, context_tokens: List[int]) -> List[int]:
        # handle segment encoding of the tokens inside the context
        token_type_ids = [
            self.CONTEXT_TOKEN_TYPE_ID for i in range(len(context_tokens))
        ]
        token_type_ids.insert(0, self.CONTEXT_TOKEN_TYPE_ID)  # bos
        return token_type_ids

    @staticmethod
    def _handle_answer_span(
            instance: IndexedInstance, raw_instance: IndexedInstance
    ):
        if "answer_position_tokenized" in raw_instance:
            ans_start, ans_end = raw_instance["answer_position_tokenized"]
            # Account for the extra BOS token in the beginning of the context
            ans_start += 1
            ans_end += 1
            instance["start_positions"] = ans_start
            instance["end_positions"] = ans_end
