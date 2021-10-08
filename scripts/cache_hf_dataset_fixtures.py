"""
Caches the tests dataset to HuggingFace's `datasets` library's cache so that the
interpreter can find it when we try to load it through the `datasets` library.
"""
from trapper.common.testing_utils.shell_utils import shell, validate_and_exit

if __name__ == "__main__":
    validate_and_exit(
        cache_test_fixtures=shell(
            "datasets-cli test test_fixtures/hf_datasets/squad_qa_test_fixture --save_infos --all_configs"
        )
    )
