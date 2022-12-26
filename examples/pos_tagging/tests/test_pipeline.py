import json

import pytest
from deepdiff import DeepDiff
from transformers import set_seed

from trapper.pipelines import create_pipeline_from_params


@pytest.fixture(scope="module")
def distilbert_conll_pipeline(experiment_params):
    set_seed(100)
    return create_pipeline_from_params(
        experiment_params,
        pipeline_type="example-pos-tagging",
    )


@pytest.fixture(scope="module")
def distilbert_pipeline_sample_input():
    return ["I love Istanbul."]


@pytest.fixture(scope="module")
def distilbert_pipeline_expected_output():
    return [
        [
            {
                "entity": "LABEL_12",
                "score": 0.035119053,
                "index": 1,
                "word": "i",
                "start": 0,
                "end": 1,
            },
            {
                "entity": "LABEL_12",
                "score": 0.036859084,
                "index": 2,
                "word": "love",
                "start": 2,
                "end": 6,
            },
            {
                "entity": "LABEL_46",
                "score": 0.03283123,
                "index": 3,
                "word": "istanbul",
                "start": 7,
                "end": 15,
            },
            {
                "entity": "LABEL_27",
                "score": 0.040444903,
                "index": 4,
                "word": ".",
                "start": 15,
                "end": 16,
            },
        ]
    ]


def test_distilbert_conll_pipeline_execution(
    distilbert_conll_pipeline,
    distilbert_pipeline_sample_input,
    distilbert_pipeline_expected_output,
):
    actual_output = distilbert_conll_pipeline(distilbert_pipeline_sample_input)
    diff = DeepDiff(
        distilbert_pipeline_expected_output,
        actual_output,
        significant_digits=3,
        ignore_numeric_type_changes=True,
    )
    assert (
        diff == {}
    ), f"Actual and Desired Dicts are not Almost Equal:\n {json.dumps(diff, indent=2)}"
