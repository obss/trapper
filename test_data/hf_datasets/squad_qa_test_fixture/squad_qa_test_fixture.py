# Copyright 2021 Open Business Software Solutions, the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test fixture for version 1.1. of SQUAD: The Stanford Question Answering Dataset.
This implementation is adapted from the question answering pipeline from the
HuggingFace's transformers library. Original code is available at:
`<https://https://github.com/huggingface/datasets/blob/master/datasets/squad/squad.py>`_.

"""

import json

import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
Small dataset taken from SQuAD v1.1 and used for testing purposes.
"""
_URL = "test_data/hf_datasets/squad_qa_test_fixture/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
}


class SquadTestFixtureConfig(datasets.BuilderConfig):
    """
    BuilderConfig for SQUAD test data.
    Args:
        **kwargs ():
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SquadTestFixture(datasets.GeneratorBasedBuilder):
    """A test dataset taken from SQUAD Version 1.1.  for trapper's QA modules"""

    BUILDER_CONFIGS = [
        SquadTestFixtureConfig(
            name="qa_test_fixture",
            version=datasets.Version("1.0.0", ""),
            description="QA test fixtures",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "paragraph_ind": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "")
                for paragraph_ind, paragraph in enumerate(article["paragraphs"]):
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"] for answer in qa["answers"]]
                        yield key, {
                            "title": title,
                            "context": context,
                            "question": qa["question"],
                            "paragraph_ind": paragraph_ind,
                            "id": qa["id"],
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
                        key += 1
