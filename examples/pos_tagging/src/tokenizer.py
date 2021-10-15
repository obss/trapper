from trapper.data import TokenizerWrapper


@TokenizerWrapper.register("pos_tagging_example", constructor="from_pretrained")
class ExamplePosTaggingTokenizerWrapper(TokenizerWrapper):
    #  Although we could have used the `TokenizerWrapper` directly, this class is
    #  implemented for demonstration purposes. You can override
    #  `_TASK_SPECIFIC_SPECIAL_TOKENS` here if your task requires custom special
    #  tokens.
    pass
