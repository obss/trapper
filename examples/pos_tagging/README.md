# POS Tagging in CONLL2003 Dataset

This project show how to train a transformer model from on CONLL2003 dataset
from `HuggingFace datasets`. You can explore the dataset
from [its page](https://huggingface.co/datasets/conll2003). This project is intended
to serve as a demo for using trapper as a library to train and evaluate a
transformer model on a custom task/dataset as well as perform inference using it. We
start by creating a fresh python environment and install the dependencies.

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

## Task Modeling

To use trapper, you need to select the common NLP formulation of the problem you are
tackling as well as decide on its input representation, including the special
tokens.

First, you need to model your problem as one of the common modeling tasks in NLP
such as seq-to-seq, sequence classification etc. We stick with the transformers' way
of dividing the tasks into common categories as it does in its `AutoModelFor...`
classes. To be compatible with transformers and reuse its model factories, trapper
formalizes the tasks by wrapping the `AutoModelFor...` classes and matching them to
a name that represent a common task in NLP. In our case, the natural choice for POS
tagging is to model it as a token classification (i.e. sequence labeling) task.

To form our input sequence, we will convert the words into tokens to get the
sequence. Then, prepend it with the special `bos_token` and append the special
`eos_token` to the end of the sequence. So the format of our instances wil
be `BOS ... tokens ... EOS`.

### Custom Class Implementations

We implement four data-related classes under `src` directory and register them to
trapper's registry. In fact, the tokenizer wrapper class is the same as the base
class, we just implemented it to show how you can implement and register a custom
tokenizer wrapper in your own tasks.

#### ExampleLabelMapperForPosTagging

Implemented in `src/label_mapper.py`. It is responsible for handling the mapping
between POS tags in CONLL2003  (such as 'FW', 'IN' etc) and their integer ids which
are fed to the model during training. Registered with the name of
`"conll2003_pos_tagging_example"` to the registry.

#### ExamplePosTaggingTokenizerWrapper

Implemented in `src/tokenizer_wrapper.py`. Although we could have used the
`TokenizerWrapper` directly, this class is implemented to demonstrate how to
subclass the TokenizerWrapper and register the new class.

#### ExampleConll2003PosTaggingDataProcessor

Implemented in `src/data_processor.py`. It extracts the `tokens`,
`pos_tags` and `id` fields from an input data instance. It tokenizes the
`tokens` field since it actually consists of words which may need further
tokenization. Then, it generates the corresponding token ids and store them.
Finally, the`pos_tags` are stored directly without any processing since this field
consists of integer labels ids instead of categorical labels.

#### ExampleDataAdapterForPosTagging

Implemented in `src/data_adapter.py`. It takes a processed instance dict from the
data processor and creates a new dict that has the `input_ids` and `labels` keys
required by the models. It also takes care of the special BOS and EOS tokens while
constructing these fields.

## Training

We encourage writing configuration files and start the training/evaluation
experiments from the trapper CLI although you can write your custom training script
as well. For this project, we will show how to use trapper using the CLI and a
config file.

### Configuration File

We use `allennlp`'s registry mechanism to provide dependency injection and enable
reading the experiment details from training configuration files in `json`
or `jsonnet` format. You can look at the
[allennlp guide on dependency injection](https://guide.allennlp.org/using-config-files)
to learn more about how the registry system and dependency injection works as well
as how to write configuration files.

In our config file, we specify all the details of an experiment, including model
type, training arguments, optimizer etc. The config file we have written is
`experiments/roberta/experiment.json`. As can be seen, we put it under `roberta`
directory and specify the model type as `"roberta-base"` (in
`pretrained_model_name_or_path` argument). You can use whatever directory structure
you like for your experiments.

#### Config File Field Explanations

`pretrained_model_name_or_path`: The pretrained model name or path from the
transformers library. This value is used for creating the pretrained model and
tokenizer inside their wrapper objects.

`train_split_name`: The split name that will be used while getting the training
dataset from the full dataset loaded via dataset reader.

`dev_split_name`: The split name that will be used while getting the validation
dataset from the full dataset loaded via dataset reader.

### trapper CLI

From the project root, go to the POS tagging project directory with the following
command:

```shell
cd examples/pos_tagging
```

Then, execute the `trapper run` command with a config file specifying the details of
the training experiment.

```shell
trapper run experiments/roberta/experiment.jsonnet
```

If you use the provided experiment config file, two directories, named `outputs`
and `results` will be created corresponding to the args["output_dir"] and
args["result_dir"] values in the config file. You can change their values to point
them to different directories, but must specify both of them (to non-existent or
empty directories). `outputs` should have the `logs` subdirectory containing the
training logs and `checkpoints` subdirectory containing the saved model weights.
The `results` directory stores the train and/or evaluation results, the trainer
state, as well as the model and the tokenizer.

## Testing Our Project

We have implemented tests checking the correctness of custom class we wrote. From
the root of the example project (i.e. the `pos_tagging` directory), you can run the
`run_tests` script as follows.

```console
cd examples/pos_tagging
python -m scripts.run_tests
```
