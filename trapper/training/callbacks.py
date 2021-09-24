from transformers import TrainerCallback as _TrainerCallback

from trapper.common import Registrable


class TrainerCallback(_TrainerCallback, Registrable):
    """
    The base class that implements `Registrable` for
    `transformers.TrainerCallback`. To add a custom callback, just subclass
    this and register the new class with a name.
    """
