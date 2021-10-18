"""
Test if the tokenizer wrapper handles correctly the general special tokens such
as BOS and EOS as well as the task-specific special tokens such as <CONTEXT>.
"""
import pytest
from transformers import PreTrainedTokenizerBase

from trapper.common.constants import ANSWER_TOKEN, CONTEXT_TOKEN
from trapper.data import TokenizerWrapper


@pytest.fixture
def bert_tokenizer_with_context_token():
    class TokenizerWrapperWithContextToken(TokenizerWrapper):
        _TASK_SPECIFIC_SPECIAL_TOKENS = [CONTEXT_TOKEN]

    return TokenizerWrapperWithContextToken.from_pretrained("bert-base-uncased")


@pytest.fixture
def gpt2_tokenizer_with_context_and_answer_tokens():
    class TokenizerWrapperWithContextAndAnswerTokens(TokenizerWrapper):
        _TASK_SPECIFIC_SPECIAL_TOKENS = [CONTEXT_TOKEN, ANSWER_TOKEN]

    return TokenizerWrapperWithContextAndAnswerTokens.from_pretrained(
        "gpt2")


def test_bert_tokenizer(bert_tokenizer_with_context_token):
    assert (bert_tokenizer_with_context_token.num_added_special_tokens
            == 1)  # CONTEXT_TOKEN
    tokenizer = bert_tokenizer_with_context_token.tokenizer
    assert tokenizer.bos_token == tokenizer.cls_token == "[CLS]"
    assert tokenizer.bos_token_id == tokenizer.cls_token_id == 101
    assert tokenizer.eos_token == tokenizer.sep_token == "[SEP]"
    assert tokenizer.sep_token_id == tokenizer.eos_token_id == 102
    assert_special_tokens_are_preserved(tokenizer, CONTEXT_TOKEN)
    assert_all_common_special_tokens_are_present(tokenizer)


def test_gpt2_tokenizer(gpt2_tokenizer_with_context_and_answer_tokens):
    assert (gpt2_tokenizer_with_context_and_answer_tokens.num_added_special_tokens
            == 4)  # PAD_TOKEN, MASK_TOKEN, CONTEXT_TOKEN, ANSWER_TOKEN.
    tokenizer = gpt2_tokenizer_with_context_and_answer_tokens.tokenizer
    assert tokenizer.bos_token == tokenizer.cls_token == "<|endoftext|>"
    assert tokenizer.bos_token_id == tokenizer.cls_token_id == 50256
    assert tokenizer.eos_token == tokenizer.sep_token == "<|endoftext|>"
    assert tokenizer.eos_token_id == tokenizer.sep_token_id == 50256
    for token in [CONTEXT_TOKEN, ANSWER_TOKEN]:
        assert_special_tokens_are_preserved(tokenizer, token)
    assert_all_common_special_tokens_are_present(tokenizer)


def assert_special_tokens_are_preserved(tokenizer: PreTrainedTokenizerBase,
                                        token: str):
    assert len(tokenizer.tokenize(token)) == 1


def assert_all_common_special_tokens_are_present(
        tokenizer: PreTrainedTokenizerBase):
    COMMON_TOKENS = (
        "bos_token", "eos_token", "cls_token", "sep_token", "pad_token",
        "mask_token", "unk_token"
    )
    assert all(hasattr(tokenizer, key) for key in COMMON_TOKENS)
