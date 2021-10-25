import pytest

from examples.pos_tagging.src import POS_TAGGING_FIXTURES_ROOT

# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
from trapper.common.testing_utils.pytest_fixtures import (
    create_data_collator_args,
    create_data_processor_args,
    get_raw_dataset,
    make_data_collator,
    make_sequential_sampler,
    temp_output_dir,
    temp_result_dir,
)

_HF_DATASETS_FIXTURES_ROOT = POS_TAGGING_FIXTURES_ROOT / "hf_datasets"


@pytest.fixture(scope="package")
def get_hf_datasets_fixture_path():
    def _get_hf_datasets_fixture_path(dataset: str) -> str:
        return str(_HF_DATASETS_FIXTURES_ROOT / dataset)

    return _get_hf_datasets_fixture_path
