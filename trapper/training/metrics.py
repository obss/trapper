from abc import ABCMeta, abstractmethod
from typing import Dict, List

import jury
import numpy as np
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.common.constants import PAD_TOKEN_LABEL_ID
from trapper.data import TransformerTokenizer


class BaseJuryMetric(Registrable, metaclass=ABCMeta):
    """
    Base `Registrable` class that is used to register the metrics needed for
    evaluating the models. The subclasses should be implemented as callables
    that accepts a `transformers.EvalPrediction` in their `__call__` method and
    compute score for that prediction.
    """
    def __init__(self, metric_name: str, label_list: List[str], tokenizer: TransformerTokenizer):
        self._jury = jury.Jury(metrics=[metric_name])
        self._label_list = label_list
        self._tokenizer = tokenizer

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        all_predicted_ids = np.argmax(pred.predictions, axis=2)
        predictions = [self._tokenizer.decode(pred_id) for pred_id in all_predicted_ids.T]
        references = self._label_list

        results = self._jury.evaluate(
                predictions=predictions, references=references
        )
        return results


@BaseJuryMetric.register("accuracy", constructor="register_accuracy")
@BaseJuryMetric.register("bertscore", constructor="register_bertscore")
@BaseJuryMetric.register("bleu", constructor="register_bleu")
@BaseJuryMetric.register("f1", constructor="register_f1")
@BaseJuryMetric.register("meteor", constructor="register_meteor")
@BaseJuryMetric.register("precision", constructor="register_precision")
@BaseJuryMetric.register("recall", constructor="register_recall")
@BaseJuryMetric.register("rouge", constructor="register_rouge")
@BaseJuryMetric.register("sacrebleu", constructor="register_sacrebleu")
@BaseJuryMetric.register("squad", constructor="register_squad")
class JuryMetric(BaseJuryMetric):
    @classmethod
    def register_accuracy(cls, *args, **kwargs):
        return cls(metric_name="accuracy", *args, **kwargs)

    @classmethod
    def register_bertscore(cls, *args, **kwargs):
        return cls(metric_name="bertscore", *args, **kwargs)

    @classmethod
    def register_bleu(cls, *args, **kwargs):
        return cls(metric_name="bleu", *args, **kwargs)

    @classmethod
    def register_f1(cls, *args, **kwargs):
        return cls(metric_name="f1", *args, **kwargs)

    @classmethod
    def register_meteor(cls, *args, **kwargs):
        return cls(metric_name="meteor", *args, **kwargs)

    @classmethod
    def register_precision(cls, *args, **kwargs):
        return cls(metric_name="precision", *args, **kwargs)

    @classmethod
    def register_recall(cls, *args, **kwargs):
        return cls(metric_name="recall", *args, **kwargs)

    @classmethod
    def register_rouge(cls, *args, **kwargs):
        return cls(metric_name="rouge", *args, **kwargs)

    @classmethod
    def register_sacrebleu(cls, *args, **kwargs):
        return cls(metric_name="sacrebleu", *args, **kwargs)

    @classmethod
    def register_squad(cls, *args, **kwargs):
        return cls(metric_name="squad", *args, **kwargs)


@BaseJuryMetric.register("seqeval")
class SeqEvalMetric(BaseJuryMetric):
    """
    Creates a token classification task metric that returns of precision,
    recall, f1 and accuracy scores. Internally, uses the "seqeval" metric
    from the HuggingFace's `datasets` library.
    Args:
        return_entity_level_metrics (bool): Set True to return all the
            entity levels during evaluation. Otherwise, returns overall
            results.
    """

    def __init__(
        self,
        label_list: List[str],
        tokenizer: TransformerTokenizer,
        return_entity_level_metrics: bool = True,
    ):
        super().__init__(
            metric_name="seqeval", label_list=label_list, tokenizer=tokenizer
        )
        self._return_entity_level_metrics = return_entity_level_metrics

    def __call__(self, pred: EvalPrediction) -> Dict[str, float]:
        all_predicted_ids = np.argmax(pred.predictions, axis=2)
        all_label_ids = pred.label_ids
        actual_predictions = []
        actual_labels = []
        for predicted_ids, label_ids in zip(all_predicted_ids, all_label_ids):
            actual_prediction = []
            actual_label = []
            for (p, l) in zip(predicted_ids, label_ids):
                if l != PAD_TOKEN_LABEL_ID:
                    actual_prediction.append(self._label_list[p])
                    actual_label.append(self._label_list[l])

            actual_predictions.append(actual_prediction)
            actual_labels.append(actual_label)

        results = self._metric.compute(
            predictions=actual_predictions, references=actual_labels
        )
        if self._return_entity_level_metrics:
            return self._extract_entity_level_metrics(results)
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def _extract_entity_level_metrics(
        self, results: Dict[str, float]
    ) -> Dict[str, float]:
        extended_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for name, val in value.items():
                    extended_results[f"{key}_{name}"] = val
            else:
                extended_results[key] = value
        return extended_results
