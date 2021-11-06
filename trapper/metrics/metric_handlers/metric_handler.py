import logging
from typing import Dict, Optional, Tuple, Union

import datasets
import numpy as np

from trapper.common import Registrable
from trapper.data import IndexedInstance
from trapper.data.label_mapper import LabelMapper
from trapper.data.tokenizers import TokenizerWrapper

logger = logging.getLogger(__file__)


class MetricHandler(Registrable):
    """
    This callable class is responsible for postprocessing evaluation output
    :py:class:`transformers.EvalPrediction` used in
    :py:class:`trapper.training.TransformerTrainer`. It is used to convert to
    suitable evaluation format for the specified metrics before metric computation.
    Do not override the `__call__()` method, instead override `postprocess()`
    methods to your task's needs and also override `extract_metadata()`
    if the task needs metadata to postprocess the evaluation output. See
    `MetricHandlerForQuestionAnswering` for an example.

    Args:
        tokenizer_wrapper (): Required to preprocess eval outputs of a model.
        label_mapper (): Only used in some tasks that require mapping between
            categorical labels and integer ids such as token classification.
    """

    default_implementation = "default"

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        label_mapper: Optional[LabelMapper] = None,
    ):
        self._tokenizer = tokenizer_wrapper
        self._label_mapper = label_mapper

    @property
    def tokenizer(self):
        return self._tokenizer.tokenizer

    @property
    def label_mapper(self):
        return self._label_mapper

    def __call__(self, dataset: datasets.Dataset) -> None:
        """
        This method applies `self.extract_metadata()` to each instance of the dataset.
        Do not override this method in child class, instead
        override `self.extract_metadata()`.

        Note:
            This handler is only called once in trainer for each dataset. By default,
            only eval_dataset is called.

        Args:
            dataset: datasets.Dataset object

        Returns: None
        """
        dataset.map(self.extract_metadata)

    def extract_metadata(self, instance: IndexedInstance) -> None:
        """
        Child class may implement this method for metadata extraction from an instance.
        It is designed to be read-only, i.e do not manipulate instance in any
        way. It should return None for efficiency purposes.

        Args:
            instance: Current instance processed

        Returns: None
        """

    def preprocess(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        references: Union[np.ndarray, Tuple[np.ndarray]],
    ) -> Tuple:
        """
        This method is called before metric computation, the default behavior is set
        in this method as directly the predicted outputs and labels. However,
        this behaviour is likely to be changed in some tasks, such as question-answering,
        etc.

        Args:
            predictions: Prediction outputs of the model.
            references: Gold labels returned.

        Returns: Preprocessed inputs.
        """
        return predictions.argmax(-1), references

    def postprocess(self, score: Dict) -> Dict:
        """
        This method is called after metric computation, the default behavior is set
        in this method as directly returning the score as is. Intended behavior of
        this method is to provide an interface to a user to manipulate score object.

        Args:
            score: Output of metric computation by `JuryMetric`.

        Returns: Post-processed score
        """
        return score


MetricHandler.register("default")(MetricHandler)
