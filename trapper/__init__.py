"""
    Exposes trapper subpackages so that they can be accessed through `trapper`
    namespace in other projects.
"""
from pathlib import Path

import trapper.common
import trapper.data
import trapper.models
import trapper.pipelines
import trapper.training
from trapper.version import VERSION as __version__  # noqa

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TESTS_ROOT = PROJECT_ROOT / "tests"
FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
