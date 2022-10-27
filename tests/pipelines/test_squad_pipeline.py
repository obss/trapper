import json

import pytest
from deepdiff import DeepDiff
from transformers import set_seed

from trapper.common.constants import SpanTuple
from trapper.common.params import Params
from trapper.pipelines import PipelineMixin


@pytest.fixture(scope="module")
def roberta_squad_pipeline_params():
    params = {
        "type": "squad-question-answering",
        "pretrained_model_name_or_path": "smallbenchnlp/roberta-small",
        "tokenizer_wrapper": {"type": "question-answering"},
        "data_processor": {"type": "squad-question-answering"},
        "data_adapter": {"type": "question-answering"},
        "data_collator": {"type": "default"},
        "model_wrapper": {"type": "question_answering"},
    }
    return Params(params)


@pytest.fixture(scope="module")
def roberta_squad_pipeline(roberta_squad_pipeline_params):
    set_seed(100)
    return PipelineMixin.from_params(roberta_squad_pipeline_params)


@pytest.fixture(scope="module")
def roberta_squad_pipeline_sample_input():
    return [
        {
            "context": 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
            "id": "0",
        }
    ]


@pytest.fixture(scope="module")
def roberta_squad_pipeline_expected_output():
    return {
        "score": 0.0002498578105587512,
        "answer": SpanTuple(
            text="Broncos defeated the National Football Conference (", start=184
        ),
    }


def test_roberta_squad_pipeline_execution(
    roberta_squad_pipeline,
    roberta_squad_pipeline_sample_input,
    roberta_squad_pipeline_expected_output,
):
    actual_output = roberta_squad_pipeline(roberta_squad_pipeline_sample_input)
    diff = DeepDiff(
        roberta_squad_pipeline_expected_output, actual_output, significant_digits=3
    )
    assert (
        diff == {}
    ), f"Actual and Desired Dicts are not Almost Equal:\n {json.dumps(diff, indent=2)}"
