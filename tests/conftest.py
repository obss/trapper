from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_TESTS_ROOT = _PROJECT_ROOT / "tests/trapper"
_FIXTURES_ROOT = _PROJECT_ROOT / "test_fixtures"


@pytest.fixture(scope="package")
def tests_root():
    return _TESTS_ROOT


@pytest.fixture(scope="package")
def fixtures_root():
    return _FIXTURES_ROOT
