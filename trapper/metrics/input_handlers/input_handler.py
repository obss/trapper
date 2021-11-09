import logging

import datasets
from transformers import EvalPrediction

from trapper.common import Registrable
from trapper.data import IndexedInstance

logger = logging.getLogger(__file__)


class MetricInputHandler(Registrable):
    """
    This callable class is responsible for processing evaluation output
    :py:class:`transformers.EvalPrediction` used in
    :py:class:`trapper.training.TransformerTrainer`. It is used to convert to
    suitable evaluation format for the specified metrics before metric computation.
    If your task needs additional information for the conversion, then override
    `self._extract_metadata()`. See
    `MetricInputHandlerForQuestionAnswering` for an example.
    """

    default_implementation = "default"

    def extract_metadata(self, dataset: datasets.Dataset) -> None:
        """
        This method applies `self._extract_metadata()` to each instance of the dataset.
        Do not override this method in child class, instead
        override `self._extract_metadata()`.

        Note:
            This method is only called once in trainer for each dataset. By default,
            only eval_dataset is called.

        Args:
            dataset: datasets.Dataset object

        Returns: None
        """
        if self._extract_metadata(dataset[0]) is not None:
            raise TypeError(
                "`_extract_metadata` method is designed to be read-only, and hence must return None."
            )
        dataset.map(self._extract_metadata)

    def _extract_metadata(self, instance: IndexedInstance) -> None:
        """
        Child class may implement this method for metadata extraction from an instance.
        It is designed to be read-only, i.e do not manipulate instance in any
        way. You can store the additional content as attributes or class variables to use
        them later in `__call__()`. It should return None for efficiency purposes.

        Args:
            instance: Current instance processed

        Returns: None
        """
        return None

    def __call__(
        self,
        eval_pred: EvalPrediction,
    ) -> EvalPrediction:
        """
        This method is called before metric computation, the default behavior is set
        in this method as returning predictions and label_ids unchanged except
        `argmax()` is applied to predictions. However, this behaviour is likely to be
        changed in some tasks, such as question-answering, etc.

        Args:
            eval_pred: EvalPrediction object returned by model.

        Returns: Processed EvalPrediction.
        """
        processsed_predictions = eval_pred.predictions.argmax(-1)
        processed_label_ids = eval_pred.label_ids
        processed_eval_pred = EvalPrediction(
            predictions=processsed_predictions, label_ids=processed_label_ids
        )
        return processed_eval_pred


MetricInputHandler.register("default")(MetricInputHandler)
