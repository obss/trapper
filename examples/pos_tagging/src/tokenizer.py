from trapper.common.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from trapper.data import TransformerTokenizer


@TransformerTokenizer.register("pos_tagging_example", constructor="from_pretrained")
class ExamplePosTaggingTokenizer(TransformerTokenizer):
    _TASK_SPECIFIC_SPECIAL_TOKENS = {
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
    }
