import pytest
from allennlp.common.util import pushd

from trapper.common.plugins import discover_plugins, import_plugins
from trapper.data import DatasetReader


@pytest.fixture(scope="module")
def plugins_root(fixtures_root):
    return fixtures_root / "plugins"


def test_no_plugins(plugins_root):
    available_plugins = set(discover_plugins())
    assert available_plugins == set()


def test_file_plugin(plugins_root):
    test_no_plugins(plugins_root)

    with pushd(plugins_root):
        available_plugins = set(discover_plugins())
        assert available_plugins == {"custom_dummy_package", "custom_dummy_module"}

        import_plugins()
        datasetreaders_available = DatasetReader.list_available()
        for name in (
            "dummy_dataset_reader_inside_package",
            "dummy_dataset_reader_inside_package2",
            "dummy_dataset_reader_inside_module",
        ):
            assert name in datasetreaders_available
