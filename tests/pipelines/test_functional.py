import pytest

from trapper.pipelines import create_pipeline_from_checkpoint


def test_create_pipeline_from_hf_hub(tmp_path):
    hf_hub_model = "albert-base-v1"
    hf_hub_model_nonexist = hf_hub_model + "abcdef"
    with pytest.raises(ValueError):
        # Must be raising an exception as:
        # "`experiment_config_path` cannot be None if `checkpoint_path` is a local_directory."
        create_pipeline_from_checkpoint(tmp_path, experiment_config_path=None)

    with pytest.raises(ValueError):
        # Must be raising an exception as:
        # "If a model is given in HF-hub, `experiment_config.json` must be included in
        # the model hub repository."
        create_pipeline_from_checkpoint(hf_hub_model)

    with pytest.raises(ValueError):
        # Must be raising an exception as:
        # "Input path must be an existing directory or an existing repository at huggingface model hub."
        create_pipeline_from_checkpoint(hf_hub_model_nonexist)
