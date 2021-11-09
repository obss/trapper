from typing import Dict

from trapper.common import Registrable


class MetricOutputHandler(Registrable):
    """
    This callable class is responsible for manipulating the resulting score object
    returned after metric computation. This base class reflects the default
    behavior as returning to result as is. See
    `MetricOutputHandlerForTokenClassification` for an example.
    """

    default_implementation = "default"

    def __call__(self, score: Dict) -> Dict:
        """
        This method is called after metric computation, the default behavior is set
        in this method as directly returning the score as is. Intended behavior of
        this method is to provide an interface to a user to manipulate the score object.

        Args:
            score: Output of metric computation by `Metric`.

        Returns: Post-processed score
        """
        return score


MetricOutputHandler.register("default")(MetricOutputHandler)
