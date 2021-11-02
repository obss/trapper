import json
import os
from copy import deepcopy
from typing import List, Dict, Union

from jury import Jury
from trapper import PROJECT_ROOT
from trapper.pipelines.pipeline import create_pipeline_from_checkpoint
# Required to register the pipeline
from trapper.pipelines.question_answering_pipeline import SquadQuestionAnsweringPipeline

from examples.question_answering.util import get_dir_from_task


def save_json(samples: List[Dict], path: str):
    with open(path, "w") as jf:
        json.dump(samples, jf)


def load_json(path: str):
    with open(path, "r") as jf:
        return json.load(jf)


def prepare_samples(data: Union[str, Dict]):
    if isinstance(data, str):
        data = load_json(data)
    data = data["data"]
    qa_samples = []

    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                sample = {}
                sample["context"] = paragraph["context"]
                sample["question"] = qa["question"]
                sample["gold_answers"] = [ans["text"] for ans in qa["answers"]]
                qa_samples.append(sample)

    return qa_samples


def prepare_samples_for_pipeline(samples: List[Dict]):
    pipeline_samples = deepcopy(samples)
    for i, sample in enumerate(pipeline_samples):
        sample.pop("gold_answers")
        if "id" not in sample:
            sample["id"] = str(i)
    return pipeline_samples


def predict(pipeline, samples, **kwargs):
    pipeline_samples = prepare_samples_for_pipeline(samples)
    predictions = pipeline(pipeline_samples, **kwargs)
    for i, prediction in enumerate(predictions):
        samples[i]["predicted_answer"] = prediction[0]["answer"].text
    return samples


def evaluate(predictions: Dict) -> Dict:
    jury = Jury(metrics="squad")
    references = [sample["gold_answers"] for sample in predictions]
    hypotheses = [sample["predicted_answer"] for sample in predictions]
    return jury.evaluate(references=references, predictions=hypotheses)


def main():
    experiment_name = "roberta-base-training-example"
    task = "question-answering"
    working_dir = os.getcwd()
    experiments_dir = os.path.join(working_dir, "experiments")
    task_dir = get_dir_from_task(os.path.join(experiments_dir, "{task}"), task=task)
    experiment_dir = os.path.join(task_dir, experiment_name)
    output_dir = os.path.join(experiment_dir, "outputs")

    pretrained_model_path = output_dir
    experiment_config = os.path.join(pretrained_model_path, "experiment_config.json")
    export_path = os.path.join(working_dir, "qa-outputs.json")
    squad_dev = os.path.join(PROJECT_ROOT, "test_fixtures/data/question_answering/squad_qa/dev.json")

    qa_pipeline = create_pipeline_from_checkpoint(
            checkpoint_path=pretrained_model_path,
            experiment_config_path=experiment_config,
            task="squad-question-answering",
            device=0
    )
    samples = prepare_samples(squad_dev)
    predictions = predict(qa_pipeline, samples)
    print(evaluate(predictions))

    # To export predictions with gold answers comment off the line below
    # save_json(predictions, export_path)


if __name__ == "__main__":
    main()
