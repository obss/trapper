# POS Tagging in CONLL 2003 Dataset
This project show how to train a transformer model from on CONLL2003 dataset 
from HuggingFace datasets. You can explore the dataset
from [its page](https://huggingface.co/datasets/conll2003).
This project is intended to serve as a demo for how to use trapper as a library 
to train and evaluate a transformer model on a custom task/dataset as well as 
perform inference using it. We start by creating a fresh python environment and 
install the dependencies in it. 

## Environment Creation and Dependency Installation

It is strongly recommended creating a virtual environment using conda or virtualenv
etc. before installing the dependencies for this example. For example, the following
code creates a conda environment with name post_tagging and python version 3.7.10,
and activates it.

```console
conda create --name post_tagging python=3.7.10
conda activate post_tagging
```

Then, you can install the dependencies using pip as follows: 

```console
pip install -r requirements.txt
```

## Modeling
To use trapper, you need to select the common NLP formulation of the problem you 
are tackling as well as its input representation including the special tokens. 

First, you need to model your problem as one of the common modeling 
tasks in NLP such as seq-to-seq, sequence classification or token classification.
We stick with the transformers' way of dividing the tasks into common categories 
as they do in their AutoModel classes. To be compatible with them and reuse 
their model factories, trapper formalizes the tasks by wrapping the `AutoModel`s 
and matching them to a name that represent a common task in NLP. In our 
case, the natural choice for POS tagging is to model it as a token classification 
(or sequence labeling) task.

To model our input, we will convert the words into tokens, prepend it 
with the special bos_token and append the special eos_token to the end of the 
sequence. So the format our instances wil be `BOS ... tokens ... EOS`.

## Training
From the project root, go to the POS tagging project directory with the following 
command:

```shell
cd examples/pos_tagging
```
Then, execute the `trapper run` command with a config file specifying the 
details of the training experiment.

```shell
trapper run experiments/roberta/experiment.jsonnet
```
If you use the provided experiment config file, two directories, named `outputs` 
and `results` will be created corresponding to the args["output_dir"] and 
args["result_dir"] values in the config file. You can change their values to point 
them to different directories, but must specify both of them (to non-existent or 
empty directories). `outputs` should contain the `logs` directory containing the 
training logs and `checkpoints` directory containing the saved model weights. 
The `results` directory stores the train and/or evaluation results, the trainer
state, as well as the model and the tokenizer.