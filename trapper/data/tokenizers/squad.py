from trapper.common.constants import CONTEXT_TOKEN
from trapper.data.tokenizers.tokenizer_wrapper import TokenizerWrapper


@TokenizerWrapper.register("question-answering", constructor="from_pretrained")
class QuestionAnsweringTokenizerWrapper(TokenizerWrapper):
    """
    This tokenizer can be used in SQuAD style question answering tasks that
    utilizes a context, question and answer.
    """

    _TASK_SPECIFIC_SPECIAL_TOKENS = [CONTEXT_TOKEN]
