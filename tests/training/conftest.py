# noinspection PyUnresolvedReferences
# pylint: disable=unused-import
import pytest

from trapper import FIXTURES_ROOT
from trapper.common.testing_utils.pytest_fixtures import (
    temp_output_dir,
    temp_result_dir,
)

HF_DATASETS_FIXTURES_PATH = FIXTURES_ROOT / "hf_datasets"


@pytest.fixture(scope="package")
def get_hf_datasets_fixture_path_from_root():
    def _get_hf_datasets_fixture_path(dataset: str) -> str:
        return str(HF_DATASETS_FIXTURES_PATH / dataset)

    return _get_hf_datasets_fixture_path

