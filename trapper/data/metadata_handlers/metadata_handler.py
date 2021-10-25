import logging
from abc import abstractmethod
from typing import Tuple, Union

import numpy as np

from trapper.common import Registrable
from trapper.data import IndexedInstance
from trapper.data.tokenizers import TokenizerWrapper

logger = logging.getLogger(__file__)


class MetadataHandler(Registrable):
    default_implementation = "default"

    def __init__(self, tokenizer_wrapper: TokenizerWrapper):
        self._tokenizer = tokenizer_wrapper.tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

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

    @abstractmethod
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
        return self.tokenizer.batch_decode(
            predictions
        ), self.tokenizer.batch_decode(references)


MetadataHandler.register("default")(MetadataHandler)
