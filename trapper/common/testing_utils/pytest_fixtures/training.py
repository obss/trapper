import pytest


@pytest.fixture(scope="module")
def temp_output_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("outputs"))


@pytest.fixture(scope="module")
def temp_result_dir(tmpdir_factory):
    return str(tmpdir_factory.mktemp("results"))
