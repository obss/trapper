# POS Tagging in CONLL2003 Dataset

This project show how to train a transformer model from on CONLL2003 dataset
from `HuggingFace datasets`. You can explore the dataset
from [its page](https://huggingface.co/datasets/conll2003). This project is intended
to serve as a demo for using trapper as a library to train and evaluate a
transformer model on a custom task/dataset as well as perform inference using it. To see an example of supported task, see [Question answering example](../question_answering). We start by creating a fresh python environment and install the dependencies.

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
tokenizer wrapper in your own tasks. For the inference, we implement a custom
pipeline class under `src/pipeline.py`.

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

#### ExamplePosTaggingPipeline

Implemented in `src/pipeline.py`. It labels the POS tags of the tokens in a given
sentence or a list of sentences.

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

`tokenizer_wrapper`: The TokenizerWrapper parameters. The generated TokenizerWrapper
object will be reused by other objects that needs it via the dependency-injection
mechanism.

`dataset_loader`: We use the base DatasetLoader class since it is sufficient for
most purposes. However, we need to specify its internal objects which we have
implemented before. These include `dataset_reader`, `data_processor` and
`data_adapter`.

`dataset_reader`: The DatasetReader construction parameters. As the argument of
`"path"` parameter, provide `conll2003_test_fixture` local variable while testing
the project whereas `"conll2003"` for the actual training.

`data_processor`: The DatasetReader construction parameters. Only specifying the
registered type is sufficient.

`data_adapter`: The DatasetAdapter construction parameters. Only specifying the
registered type is sufficient.

`data_collator`: The DataCollator construction parameters. We use the base class
which is the default registered one without any extra parameter.

`model_wrapper`: The ModelWrapper construction parameters. We specify the type of
the wrapper as  `"token_classification"` since we model the POS tagging problem as a
token classification task. Moreover, we provide a `num_labels`
argument with value of 47 as there are 47 labels in CONLL2003 POS tagging dataset.
This value will be used for modifying the output layer of the pretrained model to
make its output dimension size equal to the number of labels in the dataset if
needed.

`args`: The TransformerTrainingArguments construction parameters. Includes batch
sizes, experiment output/result and results directories, flag variables to decide on
performing train and/or evaluation and so on.

`optimizer`: The Optimizer construction parameters. We use Huggingface's AdamW
Optimizer via allennlp.

### Starting the Experiment via trapper CLI

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

We trained the `roberta-base` using this experiment configuration file on a
AWS `p3.2xlarge` instance. The training took approximately 3 hours to complete.

## Inference using pipeline object

Following code shows how to instantiate `ExamplePosTaggingPipeline` from a
checkpoint folder and use it to predict the POS tags in a sentence. You can provide
multiple sentences in the same call by putting them in a list as well. Note that
this code should be run from `examples/pos_tagging`.

```python
import src  # needed for registering the custom components
from trapper.pipelines.functional import create_pipeline_from_checkpoint
from trapper import PROJECT_ROOT

pos_tagging_project_root = PROJECT_ROOT / "examples/pos_tagging"
checkpoints_dir = (pos_tagging_project_root /
                   "outputs/roberta/outputs/checkpoints")
pipeline = create_pipeline_from_checkpoint(
        checkpoint_path=checkpoints_dir / "checkpoint-2628",
        experiment_config_path=checkpoints_dir / "experiment_config.json"
)
output = pipeline("I love Istanbul.")
print(output)
```

Note that if you used `roberta` or models based on the same tokenizer, the output
tokens contain special characters such as `Ġ`. You can write a simple
post-processing function to clean them.

## Testing Our Project

We have implemented tests checking the correctness of custom class we wrote. From
the root of the example project (i.e. the `pos_tagging` directory), you can run the
`run_tests` script as follows.

```console
cd examples/pos_tagging
export PYTHONPATH=$PYTHONPATH:$PWD
python -m scripts.run_tests
```

#### Caching the test fixtures from the HuggingFace's datasets library

To speed up the data-related tests, we cache the CONLL2003 dataset fixture from
HuggingFace's datasets library using the following command.

```console
cd examples/pos_tagging
export PYTHONPATH=$PYTHONPATH:$PWD
python -m scripts.cache_hf_datasets_fixtures
```

This can also be used to refresh the fixture dataset cache in case if it is
corrupted by any means.
