import logging
import os
import pathlib
import shutil
import tempfile

_TEMP_TEST_DIR = tempfile.mkdtemp(prefix="trapper_tests")


class TrapperTestCase:
    """
    The base testing class that stores the main project directories and takes
    care of creating and destroying a temp directory as a test fixture.
    """

    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    MODULE_ROOT = PROJECT_ROOT / "trapper"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG
        )
        self.TEST_DIR = pathlib.Path(_TEMP_TEST_DIR)
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
