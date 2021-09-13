import os
import warnings
from typing import Tuple

import requests

WORKING_DIR = os.path.abspath(os.path.dirname(__file__))
EXPERIMENTS_DIR = os.path.join(WORKING_DIR, "experiments")
TASK_DIR = os.path.join(EXPERIMENTS_DIR, "{task}")
DATASET_DIR = os.path.join(TASK_DIR, "datasets")
MODEL_DIR = os.path.join(TASK_DIR, "model")
CHECKPOINT_DIR = os.path.join(TASK_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(TASK_DIR, "outputs")

DEFAULT_EXTRA_VARIABLES = {
	"TRAIN_DATA_PATH": os.path.join(DATASET_DIR, "train.json"),
	"DEV_DATA_PATH": os.path.join(DATASET_DIR, "dev.json"),
	"OUTPUT_PATH": OUTPUT_DIR,
	"CHECKPOINT_PATH": CHECKPOINT_DIR
}


def download_squad(task: str, version: str = "1.1", overwrite: bool = False) -> Tuple[str, str]:
	"""
	Downloads SQuAD dataset with given version.

	Args:
		version: SQuAD dataset version.
		overwrite: If true, overwrites the destination file.

	Returns: (train set path, dev set path) local paths of downloaded dataset files.

	"""
	destination_dir = DATASET_DIR.format(task=task)
	os.makedirs(destination_dir, exist_ok=True)
	dataset_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

	train_set = f"train-v{version}.json"
	dev_set = f"dev-v{version}.json"

	datasets = [train_set, dev_set]
	paths = []

	for dataset in datasets:
		url = os.path.join(dataset_base_url, dataset)
		dest = os.path.join(destination_dir, dataset)
		paths.append(dest)

		if not overwrite and os.path.exists(dest):
			warnings.warn(f"{dest} already exists, not overwriting.")
			continue

		r = requests.get(url, allow_redirects=True)

		with open(dest, 'wb') as out_file:
			out_file.write(r.content)

	return paths[0], paths[1]


def create_dir(prefix: str, name: str):
	path = os.path.join(prefix, name)
	os.makedirs(path, exist_ok=True)
	return path


def get_dir_from_task(path: str, task: str):
	task = "unnamed-task" if task is None else task
	return path.format(task=task)
