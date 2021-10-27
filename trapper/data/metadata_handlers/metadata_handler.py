import logging
from abc import abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from trapper.common import Registrable
from trapper.data import IndexedInstance
from trapper.data.label_mapper import LabelMapper
from trapper.data.tokenizers import TokenizerWrapper

logger = logging.getLogger(__file__)


class MetadataHandler(Registrable):
    """
    This callable class is responsible for postprocessing evaluation output
    :py:class:`transformers.EvalPrediction` used in
    :py:class:`trapper.training.TransformerTrainer`. It is used to convert to
    suitable evaluation format for the specified metrics before metric computation.
    Do not override the `__call__()` method, instead override `postprocess()`
    methods to your task's needs and also override `extract_metadata()`
    if the task needs metadata to postprocess the evaluation output. See
    `MetadataHandlerForQuestionAnswering` for an example.

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

    def __call__(self, instance: IndexedInstance, split: str) -> IndexedInstance:
        """
        Where extraction from instances happen through an abstractmethod
        extract_metadata, returns the input instance as is. This method cannot be
        overridden in child classes.

        Args:
            instance: Indexed instance.
            split: split of the data.

        Returns: IndexedInstance as is.
        """
        # Currently through HF trainer validation split is used for eval
        if split == "validation":
            self.extract_metadata(instance)
        return instance

    def extract_metadata(self, instance: IndexedInstance) -> None:
        """
        All child class must implement this method for metadata extraction from instance.

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


MetadataHandler.register("default")(MetadataHandler)