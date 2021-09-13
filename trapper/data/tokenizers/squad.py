from trapper.common.constants import BOS_TOKEN, CONTEXT_TOKEN, EOS_TOKEN, PAD_TOKEN
from trapper.data.tokenizers.tokenizer import TransformerTokenizer


@TransformerTokenizer.register("question-answering", constructor="from_pretrained")
class QuestionAnsweringTokenizer(TransformerTokenizer):
    """
    This tokenizer can be used in SQuAD style question answering tasks that
    utilizes a context, question and answer.
    """

    _attr_to_special_token = {"additional_special_tokens": [CONTEXT_TOKEN]}
    _TASK_SPECIFIC_SPECIAL_TOKENS = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
    }
