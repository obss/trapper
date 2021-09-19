import shutil
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_TESTS_ROOT = _PROJECT_ROOT / "tests/trapper"
_FIXTURES_ROOT = _PROJECT_ROOT / "test_data"


@pytest.fixture(scope="package")
def test_root():
    return _TESTS_ROOT


@pytest.fixture(scope="package")
def fixtures_root():
    return _FIXTURES_ROOT


@pytest.fixture
def tempdir():
    dir_path = Path(tempfile.mkdtemp(prefix="test"))
    yield dir_path
    shutil.rmtree(dir_path)
