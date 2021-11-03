import os

from trapper.common.notebook_utils.file_transfer import download_from_url

FIXTURES_PATH = "squad_qa_test_fixture"
SQUAD_QA_FIXTURES = {
    "dev.json": "https://raw.githubusercontent.com/obss/trapper/main/test_fixtures/hf_datasets/squad_qa_test_fixture/dev.json",
    "train.json": "https://raw.githubusercontent.com/obss/trapper/main/test_fixtures/hf_datasets/squad_qa_test_fixture/train.json",
    "squad_qa_test_fixture.py": "https://raw.githubusercontent.com/obss/trapper/main/test_fixtures/hf_datasets/squad_qa_test_fixture/squad_qa_test_fixture.py",
}
EXPERIMENT_CONFIG = "https://raw.githubusercontent.com/obss/trapper/main/examples/question_answering/experiment.jsonnet"


def download_fixture_data():
    for file, url in SQUAD_QA_FIXTURES.items():
        destination = os.path.join(FIXTURES_PATH, file)
        download_from_url(url, destination)


def fetch_experiment_config():
    download_from_url(EXPERIMENT_CONFIG)


def prepare_data():
    download_fixture_data()
    fetch_experiment_config()
