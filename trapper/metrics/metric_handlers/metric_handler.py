import logging
from typing import Optional, Tuple, Union

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
        tokenizer_wrapper (): Required to postprocess eval outputs of a model.
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

    @tokenizer.setter
    def tokenizer(self, value):
        if isinstance(value, TokenizerWrapper):
            self._tokenizer = value
        else:
            raise ValueError(
                f"The value must be an instance of a "
                f"class derived from {TokenizerWrapper}"
            )

    @property
    def label_mapper(self):
        return self._label_mapper

    @label_mapper.setter
    def label_mapper(self, value):
        if isinstance(value, LabelMapper):
            self._label_mapper = value
        else:
            raise ValueError(
                f"The value must be an instance of a "
                f"class derived from {LabelMapper}"
            )

    def __call__(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Where extraction from instances happen through the method
        `extract_metadata()`, returns the input dataset as is. Do not override
        this method in child class, instead use `MetricHandler.extract_metadata()`.

        Args:
            dataset:

        Returns: IndexedInstance as is.
        """
        # Currently through HF trainer validation split is used for eval
        dataset.map(self.extract_metadata)
        return dataset

    def extract_metadata(self, instance: IndexedInstance) -> None:
        """
        Child class may implement this method for metadata extraction from an instance.

        Args:
            instance: Current instance processed

        Returns: None
        """
        pass

    def postprocess(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        references: Union[np.ndarray, Tuple[np.ndarray]],
    ) -> Tuple:
        """
        This method is called before metric computation, the default behavior is set
        in this method as directly decoding the predicted outputs and labels. However,
        this behaviour is likely to be changed in some tasks, such as question-answering,
        etc.

        Args:
            predictions:
            references:

        Returns: Post-processed inputs.
        """
        predictions = predictions.argmax(-1)
        return self.tokenizer.batch_decode(
            predictions
        ), self.tokenizer.batch_decode(references)


MetricHandler.register("default")(MetricHandler)
