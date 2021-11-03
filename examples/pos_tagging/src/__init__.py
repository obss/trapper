from pathlib import Path

from examples.pos_tagging.src import data
from examples.pos_tagging.src.metric_handler import MetricHandlerForPosTagging
from examples.pos_tagging.src.pipeline import ExamplePosTaggingPipeline

POS_TAGGING_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
POS_TAGGING_TESTS_ROOT = POS_TAGGING_PROJECT_ROOT / "tests"
POS_TAGGING_FIXTURES_ROOT = POS_TAGGING_PROJECT_ROOT / "test_fixtures"
