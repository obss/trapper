"""
Caches the tests dataset to HuggingFace's `datasets` library's cache so that the
interpreter can find it when we try to load it through the `datasets` library.
"""
from examples.pos_tagging.src import POS_TAGGING_FIXTURES_ROOT
from trapper.common.testing_utils.hf_datasets_caching import (
    renew_hf_datasets_fixtures_cache,
)

if __name__ == "__main__":
    renew_hf_datasets_fixtures_cache(POS_TAGGING_FIXTURES_ROOT / "hf_datasets")
