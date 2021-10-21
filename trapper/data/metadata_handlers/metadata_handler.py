import logging
from abc import abstractmethod
from typing import final, Tuple

import numpy as np

from trapper.common import Registrable
from trapper.data import IndexedInstance
from trapper.data.tokenizers import TransformerTokenizer

logger = logging.getLogger(__file__)


class MetadataHandler(Registrable):
    default_implementation = "default"

    def __init__(self, tokenizer: TransformerTokenizer):
        self._tokenizer = tokenizer

    @final
    def __call__(
        self, instance: IndexedInstance, split: str
    ) -> IndexedInstance:
        """
        Where extraction from instances happen through an abstractmethod
        extract_metadata, returns the input instance as is. This method cannot be
        overridden in child classes.

        Args:
            instance: Indexed instance.
            split: split of the data.

        Returns: IndexedInstance as is.
        """
        self.extract_metadata(instance, split)
        return instance

    @abstractmethod
    def extract_metadata(self, instance: IndexedInstance, split: str) -> None:
        """
        All child class must implement this method for metadata extraction from instance.

        Args:
            instance: Current instance processed

        Returns: None
        """
        pass

    def postprocess(self, predictions: Tuple[np.ndarray], references: Tuple[np.ndarray]) -> Tuple:
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
        return self._tokenizer.batch_decode(predictions), self._tokenizer.batch_decode(references)


MetadataHandler.register("default")(MetadataHandler)
