import os
import shutil
from pathlib import Path

from trapper.common.constants import Pathlike
from trapper.common.testing_utils.shell_utils import shell, validate_and_exit


def get_hf_cache_dir() -> Path:
    return Path(
        os.environ.get("HF_DATASETS_CACHE") or Path.home() / ".cache/huggingface"
    )


def remove_hf_datasets_fixtures_cache(hf_datasets_fixtures_path: Pathlike) -> None:
    hf_datasets_fixtures_path = Path(hf_datasets_fixtures_path)
    hf_cache_dir = get_hf_cache_dir()
    hf_cached_datasets_dir = hf_cache_dir / "datasets"
    hf_cached_dataset_modules_dir = (
        hf_cache_dir / "modules/datasets_modules/datasets"
    )

    for fixture_dataset in hf_datasets_fixtures_path.glob("*"):
        # Remove from the original fixture directory
        try:
            os.remove(fixture_dataset / "dataset_infos.json")
            for f in fixture_dataset.glob("*.lock"):
                os.remove(f)
        except:
            pass

        # Remove from the global HuggingFace dataset cache
        for p in hf_cached_datasets_dir.glob(f"*{fixture_dataset.name}*"):
            if os.path.isfile(p):
                os.remove(p)
            shutil.rmtree(p, ignore_errors=True)

        # Remove from the global HuggingFace datasets modules cache
        shutil.rmtree(
            hf_cached_dataset_modules_dir / f"{fixture_dataset.name}",
            ignore_errors=True,
        )


def cache_hf_datasets_fixtures(hf_datasets_fixtures_path: Pathlike) -> None:
    hf_datasets_fixtures_path = str(hf_datasets_fixtures_path)
    commands = {}
    for d in os.listdir(hf_datasets_fixtures_path):
        commands[f"cache_{d}"] = shell(
            f"datasets-cli test {hf_datasets_fixtures_path}/{d} --save_infos --all_configs"
        )
    validate_and_exit(**commands)


def renew_hf_datasets_fixtures_cache(hf_datasets_fixtures_path: Pathlike) -> None:
    remove_hf_datasets_fixtures_cache(hf_datasets_fixtures_path)
    cache_hf_datasets_fixtures(hf_datasets_fixtures_path)
