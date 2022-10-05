import numpy as np
from transformers import EvalPrediction

from trapper.data.tokenizers import TokenizerWrapper
from trapper.metrics.input_handlers import MetricInputHandler


@MetricInputHandler.register("language-generation")
class MetricInputHandlerForLanguageGeneration(MetricInputHandler):
    """
    `MetricInputHandlerForLanguageGeneration` provides the conversion from token ids
    to decoded strings for predictions and labels and prepares them for the metric
    computation.

    Args:
        tokenizer_wrapper (): Required to convert token ids to strings.
    """

    _contexts = list()

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        super(MetricInputHandlerForLanguageGeneration, self).__init__()
        self._tokenizer_wrapper = tokenizer_wrapper

    @property
    def tokenizer(self):
        return self._tokenizer_wrapper.tokenizer

    def preprocess(self, eval_pred: EvalPrediction) -> EvalPrediction:
        if isinstance(eval_pred.predictions, tuple):
            eval_pred = EvalPrediction(
                # Models like T5 returns a tuple of
                # (logits, encoder_last_hidden_state) instead of only the logits
                predictions=eval_pred.predictions[0],
                label_ids=eval_pred.label_ids,
            )
        eval_pred = super(MetricInputHandlerForLanguageGeneration, self).preprocess(
            eval_pred
        )

        # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/examples/pytorch/translation/run_translation.py#L540
        references = np.where(
            eval_pred.label_ids != -100,
            eval_pred.label_ids,
            self.tokenizer.pad_token_id,
        )

        # Batch decode is intentionally avoided as jury metrics expect
        # list of list of string for language-generation metrics.
        predictions = np.array(
            [
                [self.tokenizer.decode(pred, skip_special_tokens=True)]
                for pred in eval_pred.predictions
            ]
        )
        references = np.array(
            [
                [self.tokenizer.decode(ref, skip_special_tokens=True)]
                for ref in references
            ]
        )

        return EvalPrediction(predictions=predictions, label_ids=references)
