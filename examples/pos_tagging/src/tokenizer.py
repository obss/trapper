from trapper.data import TokenizerFactory


@TokenizerFactory.register("pos_tagging_example", constructor="from_pretrained")
class ExamplePosTaggingTokenizerFactory(TokenizerFactory):
    #  Although we could have used the `TokenizerFactory` directly, this class is
    #  implemented for demonstration purposes. You can override
    #  `_TASK_SPECIFIC_SPECIAL_TOKENS` here if your task requires custom special
    #  tokens.
    pass
