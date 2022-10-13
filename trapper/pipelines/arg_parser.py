from transformers.pipelines import ArgumentHandler as _ArgumentHandler

from trapper.common import Registrable


class ArgumentHandler(_ArgumentHandler, Registrable):
    """
    Registered ArgumentHandler class for pipeline class/subclasses.
    """


ArgumentHandler.register("default")(ArgumentHandler)
