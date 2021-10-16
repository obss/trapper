# TRAPPER (Transformer wRAPPER)

A lightweight framework that aims to make it easier to train transformer based
models on downstream tasks. It wraps the `HuggingFace`'s
`transformers` library to provide the transformer model implementations and training
mechanisms conveniently.

* `allennlp`'s registry mechanism is used to provide dependency injection and enable
  reading the experiment details from training configuration files which are
  typically `json` or `jsonnet` files. Moreover, registrable base classes are
  implemented to abstract away the common operations for data processing and model
  training.

* Auto classes from `transformers` are used to provide polymorphism and make it
  possible to instantiate the actual task-specific classes (e.g. for models and
  tokenizers) from the configuration files dynamically.

## Table of Currently Supported Tasks and Models From Transformers

| model       | question_answering | token_classification |
|-------------|--------------------|----------------------|
| ALBERT      | &#10004;           | &#10004;             |
| DistillBERT | &#10004;           | &#10004;             |
| ELECTRA     | &#10004;           | &#10004;             |
| RoBERTa     | &#10004;           | &#10004;             |

## Usage

To use trapper on training, evaluation on a task that is not readily supported in
transformers library, you need to extend the provided base classes according to 
your own needs. These are as follows:

**For Training & Evaluation**: DataProcessor, DataAdapter, TokenizerFactory.

**For Inference**: Additionally, you may need to implement a `transformers.
Pipeline` or directly use form the transformers library if they already implemented
one that matches your need.

1) **DataProcessor**:
This class is responsible for taking a single instance in dict format, typically 
   coming from a `datasets.Dataset`, extracting the information fields suitable 
   for the task and hand, and converting their values to integers or collections 
   of integers. This includes, tokenizing the string fields, and getting the 
   token ids, converting the categoric labels to integer ids and so on.
   

2) **DataAdapter**:
This is responsible for converting the information fields inside an instance 
   dict that was previously processed by a `DataProcessor` to a format suitable 
   for feeding into a transformer model. This also includes handling the special tokens
   signaling the start or end of a sequence, the separation of tho sequence for 
   a sequence-pair task as well as chopping excess tokens etc.

3) **TokenizerWrapper**:
This class wraps a pretrained tokenizer from the transformers library while 
   also recording the special tokens needed for the task to the tokenizer. It also
#### Registering classes from custom modules to the library

We support both file based and command line argument based approaches to register
the external modules written by the users.

##### Option 1 - File based

You should list the packages or modules (for stand-alone modules not residing inside
any package) containing the classes to be registered as plugins to a local file
named `.trapper_plugins`. This file must reside in the same directory where you run
the `trapper run` command. Moreover, it is recommended to put the plugins file where
the modules to be registered resides (e.g. the project root) for convenience since
that directory will be added to the `PYTHONPATH`. Otherwise, you need to add the
plugin module/packages to the `PYTHONPATH` manually. Another reminder is that each
listed package must have an `__init__.py` file that imports the modules containing
the custom classes to be registered.

E.g., let's say our project's root directory is `project_root` and the experiment
config file is inside the root with a name `test_experiment.jsonnet`. To run the
experiment, you should run the following commands:

```shell
cd project_root
trapper run test_experiment.jsonnet
```

Below output shows the content of the project_root directory.

```console
ls project_root

  ner
  tests
  datasets
  .trapper_plugins
  test_experiment.jsonnet
```

Additionally, here is the content of the project_root/.trapper_plugins.

```console
cat project_root/.trapper_plugins

  ner.core.models
  ner.data.dataset_readers
```

##### Option 2 - Using the command line argument

You can specify the packages and/or modules you want to be registered using the
--include-package argument. However, note that you need to repeat the argument for
each package/module to be registered.

E.g. the running the following commands is an alternative to `Option-1` to start the
experiment specified in the `test_experiment.jsonnet`.

```console
trapper run test_experiment.jsonnet \
--include-package ner.core.models \
--include-package ner.data.dataset_readers
```

#### Running a training and/or evaluation experiment

##### Config File Based Training Using the CLI

Go to your project root and execute the `trapper run` command with a config file
specifying the details of the training experiment. E.g.

```shell
trapper run SOME_DIRECTORY/experiment.jsonnet
```

Don't forget to provide the args["output_dir"] and args["result_dir"] values in your
experiment file. Please look at the `examples/pos_tagging/README.md` for a detailed
example.

##### Scrip Based Training

Go to your project root and execute the `trapper run` command with a config file
specifying the details of the training experiment. E.g.

```shell
trapper run SOME_DIRECTORY/experiment.jsonnet
```

Don't forget to provide the args["output_dir"] and args["result_dir"] values in your
experiment file. Please look at the `examples/pos_tagging/README.md` for a detailed
example.

## Contributing

PRs are always welcome :)

### Installation

#### Environment Creation

It is strongly recommended creating a virtual environment using conda or virtualenv
etc. before installing this package and its dependencies. For example, the following
code creates a conda environment with name trapper and python version 3.7.10, and
activates it.

```console
conda create --name trapper python=3.7.10
conda activate trapper
```

Then, you can install trapper and its dependencies from either the main branch as
show below or any other branch / tag / commit you would like.

#### Regular Installation

(WIP - currently, repo is private)

```console
pip install git+ssh://github.com/obss/trapper.git
```

#### Installing in Editable Mode

You can clone the repo locally and install the package to your environment as shown
below.

```console
git clone https://github.com/obss/trapper.git
cd trapper
pip install -e .[dev]
```

### Testing trapper

#### Caching the test fixtures from the HuggingFace's datasets library

To speed up the data-related tests and enable accessing the fixture datasets by
their folder name, we cache the test dataset fixtures from HuggingFace's datasets
library using the following command.

```console
python -m scripts.cache_hf_datasets_fixtures
```

Then, you can simply test with the following command:

```console
python -m scripts.run_tests
```

**NOTE:** To significantly speed up the tests, you can set the following environment
variables which makes HuggingFace's transformers and datasets libraries work in
offline mode. However, beware that you may need to run the tests once first without
setting these environment variables so that the models, tokenizers etc. are cached.

```shell
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

### Code Style

To check code style,

```console
python -m scripts.run_code_style check
```

To format codebase,

```console
python -m scripts.run_code_style format
```

## Contributors

- [Cemil Cengiz](https://github.com/cemilcengiz)
- [Devrim Çavuşoğlu](https://github.com/devrimcavusoglu)