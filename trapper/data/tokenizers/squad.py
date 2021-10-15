from trapper.common.constants import CONTEXT_TOKEN
from trapper.data.tokenizers.tokenizer import TokenizerFactory


@TokenizerFactory.register("question-answering", constructor="from_pretrained")
class QuestionAnsweringTokenizerFactory(TokenizerFactory):
    """
    This tokenizer can be used in SQuAD style question answering tasks that
    utilizes a context, question and answer.
    """
    _TASK_SPECIFIC_SPECIAL_TOKENS = [CONTEXT_TOKEN]
