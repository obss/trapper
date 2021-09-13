from dataclasses import dataclass, field
from typing import Optional

from transformers.training_args import TrainingArguments as _TrainingArguments
from transformers.utils import logging

from trapper.common import Registrable
from trapper.common.utils import append_parent_docstr

logger = logging.get_logger(__name__)


@append_parent_docstr(parent_id=0)
@dataclass
class TransformerTrainingArguments(_TrainingArguments, Registrable):
    """
    Wraps the `TrainingArguments` class from the `Transformers` library.
    """

    result_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to save the metrics for the final model "
            "and the trainer state at the end of the training."
        },
    )

    def __post_init__(self):
        if self.report_to is None:
            logger.info(
                "Transformers v4.5.1 defaults `--report_to` to 'all', "
                "so we change it to 'tensorboard'."
            )
            self.report_to = ["tensorboard"]
        super().__post_init__()


TransformerTrainingArguments.register("default")(TransformerTrainingArguments)
