import os

import pytest
from deepdiff import DeepDiff
from transformers import set_seed

from trapper import FIXTURES_ROOT
from trapper.common.constants import SpanTuple
from trapper.pipelines import create_pipeline_from_checkpoint
from trapper.training.train import run_experiment

PIPELINE_FIXTURES = FIXTURES_ROOT / "pipelines"

@pytest.fixture(scope="module")
def integration_expected_training_result():
    return {'epoch': 10.0,
    'eval_empty_items': 0,
    'eval_loss': 5.136352062225342,
    'eval_total_items': 6}

@pytest.fixture(scope="module")
def integration_single_inference_input():
    return [
        {'context': 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',
        'question': 'Which NFL team represented the AFC at Super Bowl 50?',
        'id': '0'}
    ]

@pytest.fixture(scope="module")
def integration_multi_inference_input():
    return [
        {'context': 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',
        'question': 'Which NFL team represented the AFC at Super Bowl 50?',
        'id': '0'},
        {'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
        'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        'id': '1'}
    ]

@pytest.fixture(scope="module")
def integration_expected_single_inference():
    return {
        'answer': SpanTuple(text='would have been known as', start=667),
        'score': 0.00032179668778553605
    }

@pytest.fixture(scope="module")
def integration_expected_multi_inference():
    return [
        {'answer': SpanTuple(text='would have been known as', start=667),
        'score': 0.00032179668778553605},
        {'answer': SpanTuple(text='end of the main drive (and in a direct line that connects', start=558),
        'score': 0.0003353202191647142}
    ]

def test_integration(
        tmp_path,
        integration_expected_training_result,
        integration_single_inference_input,
        integration_expected_single_inference,
        integration_multi_inference_input,
        integration_expected_multi_inference
    ):
    set_seed(100)
    experiment_dir = tmp_path
    config_path = FIXTURES_ROOT / "pipelines/pipeline_integration_experiment.jsonnet"

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    output_dir = os.path.join(experiment_dir, "outputs")

    ext_vars = {
        # Used to feed the jsonnet config file with file paths
        "OUTPUT_PATH": output_dir,
        "CHECKPOINT_PATH": checkpoint_dir
    }

    result = run_experiment(
        config_path=str(config_path),
        ext_vars=ext_vars,
    )

    for k in integration_expected_training_result.keys():
        assert integration_expected_training_result[k] <= result[k]

    PRETRAINED_MODEL_PATH = output_dir
    EXPERIMENT_CONFIG = os.path.join(PRETRAINED_MODEL_PATH, "experiment_config.json")

    qa_pipeline = create_pipeline_from_checkpoint(
        checkpoint_path=PRETRAINED_MODEL_PATH,
        experiment_config_path=EXPERIMENT_CONFIG,
        pipeline_type="squad-question-answering"
    )

    actual_single_inference = qa_pipeline(integration_single_inference_input)

    diff = DeepDiff(integration_expected_single_inference, actual_single_inference , significant_digits=3)
    assert not diff, "Single Inference Results are not as Expected:"

    actual_multi_inference = qa_pipeline(integration_multi_inference_input)

    diff = DeepDiff(integration_expected_multi_inference, actual_multi_inference, significant_digits=3)
    assert not diff, "Multiple Inference Results are not as Expected:"
