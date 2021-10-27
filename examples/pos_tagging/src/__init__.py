from pathlib import Path

from examples.pos_tagging.src.data_adapter import ExampleDataAdapterForPosTagging
from examples.pos_tagging.src.data_processor import (
    ExampleConll2003PosTaggingDataProcessor,
)
from examples.pos_tagging.src.tokenizer_wrapper import (
    ExamplePosTaggingTokenizerWrapper,
)
from examples.pos_tagging.src.metadata_handler import MetadataHandlerForPosTagging

POS_TAGGING_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
POS_TAGGING_TESTS_ROOT = POS_TAGGING_PROJECT_ROOT / "tests"
POS_TAGGING_FIXTURES_ROOT = POS_TAGGING_PROJECT_ROOT / "test_fixtures"
