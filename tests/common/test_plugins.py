from allennlp.common.util import pushd
from overrides import overrides

from trapper.common.plugins import discover_plugins, import_plugins
from trapper.common.testing.test_case import TrapperTestCase
from trapper.data import DatasetLoader


class TestPlugins(TrapperTestCase):
    @overrides
    def setup_method(self):
        super().setup_method()
        self.plugins_root = self.FIXTURES_ROOT / "plugins"

    def test_no_plugins(self):
        available_plugins = set(discover_plugins())
        assert available_plugins == set()

    def test_file_plugin(self):
        self.test_no_plugins()

        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            assert available_plugins == {"custom_dummy_package",
                                         "custom_dummy_module"}

            import_plugins()
            dataset_loaders_available = DatasetLoader.list_available()
            for name in (
                    "dummy_dataset_loader_inside_package",
                    "dummy_dataset_loader_inside_package2",
                    "dummy_dataset_loader_inside_module",
            ):
                assert name in dataset_loaders_available
