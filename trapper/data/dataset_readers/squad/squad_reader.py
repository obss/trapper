from abc import ABCMeta
from typing import Dict, List, Optional, Union

from trapper.common.constants import PositionTuple, SpanTuple
from trapper.data.dataset_readers.dataset_reader import BasicDatasetReader


class SquadDatasetReader(BasicDatasetReader, metaclass=ABCMeta):
    @staticmethod
    def _join_whitespace_prefix(context: str, field: SpanTuple) -> SpanTuple:
        """
        Prepend the whitespace prefix if it exists. Some tokenizers like
        `roberta` and `gpt2` does not ignore the whitespaces, which leading to
        the difference in the token ids between directly tokenizing the field or
        tokenizing it with the context.
        Args:
            context (str): context e.g. paragraph or document.
            field (SpanTuple): a special segment of the context such as
                `answer` or `clue` in the Question Generation task.

        Returns:
        """
        start = getattr(field, "start", None)
        if start is not None and context[start - 1] == " ":
            return SpanTuple(text=" " + field.text, start=start - 1)
        return field

    def _store_tokenized_field_position(
        self, instance, context: str, start: int, type_: str
    ):
        instance[type_ + "_position_tokenized"] = self._tokenized_field_position(
            context, instance["context"], instance[type_], start
        )

    def _tokenized_field_position(
        self,
        context: str,
        context_token_ids: List[int],
        field_token_ids: List[int],
        start: int,
    ) -> PositionTuple:
        tokenized_prefix = self._tokenizer.tokenize(context[:start])
        prefix_ids = self._tokenizer.convert_tokens_to_ids(tokenized_prefix)
        return self._get_position(context_token_ids, field_token_ids, prefix_ids)

    @staticmethod
    def _get_position(context_ids, field_ids, field_prefix_ids) -> PositionTuple:
        """Returns the start and end indices of the field (e.g. answer or
        clue) span in the paragraph (context)"""
        diff_ind = min(len(field_prefix_ids), len(context_ids))
        # Find the first token where the context_ids and field_prefix ids differ
        for i, (context_id, field_prefix_id) in enumerate(
            zip(context_ids, field_prefix_ids)
        ):
            if context_id != field_prefix_id:
                diff_ind = i
                break
        return PositionTuple(
            diff_ind, min(diff_ind + len(field_ids), len(context_ids))
        )

    def _indexed_field(
        self,
        context: str,
        context_token_ids: List[int],
        field: SpanTuple,
        field_type: str,
    ) -> Optional[Dict[str, Union[List[int], PositionTuple]]]:
        field_tokens = self._tokenizer.tokenize(field.text)
        field_token_ids = self._tokenizer.convert_tokens_to_ids(field_tokens)
        indexed_field = {field_type: field_token_ids}
        if field.start is not None:
            field_position = self._tokenized_field_position(
                context, context_token_ids, field_token_ids, field.start
            )
            if field_position.start + len(field_tokens) > len(context_token_ids):
                return None
            indexed_field[f"{field_type}_position_tokenized"] = field_position
        return indexed_field

    def _chop_excess_context_tokens(
        self, tokenized_context: List, *other_tokenized_subsequences: List
    ):
        subsequences = [tokenized_context, *other_tokenized_subsequences]
        seq_len = self._total_seq_len(*subsequences)
        if seq_len > self._tokenizer.model_max_sequence_length:
            self._chop_excess_tokens(tokenized_context, seq_len)
            seq_len = self._total_seq_len(*subsequences)
            assert seq_len <= self._tokenizer.model_max_sequence_length
