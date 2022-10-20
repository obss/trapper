import json

import pytest
from deepdiff import DeepDiff
from transformers import set_seed
from trapper import PROJECT_ROOT
from trapper.common.constants import SpanTuple
from trapper.pipelines import create_pipeline_from_params

from examples.pos_tagging.src.pipeline import ExamplePosTaggingPipeline


@pytest.fixture(scope="function")
def model_checpoint_dir():
    pos_tagging_project_root = PROJECT_ROOT / "examples/pos_tagging"
    checkpoints_dir = (pos_tagging_project_root /
                       "outputs/roberta/outputs/checkpoints")
    return checkpoints_dir


@pytest.fixture(scope="module")
def distilbert_conll_pipeline(experiment_params):
    set_seed(100)
    return create_pipeline_from_params(
            experiment_params,
            pipeline_type="example-pos-tagging",
            pretrained_model_name_or_path="distilbert-base-uncased"
    )


@pytest.fixture(scope="module")
def distilbert_pipeline_sample_input():
    return [
        "I love Istanbul."
    ]


@pytest.fixture(scope="module")
def distilbert_pipeline_expected_output():
    return {
        "score": 0.0002498578105587512,
        "answer": SpanTuple(
            text="Broncos defeated the National Football Conference (", start=184
        ),
    }


def test_roberta_squad_pipeline_execution(
    distilbert_conll_pipeline,
    distilbert_pipeline_sample_input,
    distilbert_pipeline_expected_output,
):
    actual_output = distilbert_conll_pipeline(distilbert_pipeline_sample_input)
    diff = DeepDiff(
        actual_output, distilbert_pipeline_expected_output, significant_digits=3
    )
    assert (
        diff == {}
    ), f"Actual and Desired Dicts are not Almost Equal:\n {json.dumps(diff, indent=2)}"
