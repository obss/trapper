"""
    Exposes trapper subpackages so that they can be accessed through `trapper`
    namespace in other projects.
"""
import trapper.common
import trapper.data
import trapper.models
import trapper.pipelines
import trapper.training
from trapper.version import VERSION as __version__  # noqa
