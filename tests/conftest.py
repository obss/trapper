from pathlib import Path

import pytest

from trapper import FIXTURES_ROOT, TESTS_ROOT

_HF_DATASETS_FIXTURES_ROOT = FIXTURES_ROOT / "hf_datasets"


@pytest.fixture(scope="package")
def tests_root():
    return TESTS_ROOT


@pytest.fixture(scope="package")
def fixtures_root() -> Path:
    return FIXTURES_ROOT


@pytest.fixture(scope="package")
def get_hf_datasets_fixture_path():
    def _get_hf_datasets_fixture_path(dataset: str) -> str:
        return str(_HF_DATASETS_FIXTURES_ROOT / dataset)

    return _get_hf_datasets_fixture_path
