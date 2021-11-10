<h1 align="center">Trapper (Transformers wRAPPER)</h1>

<p align="center">
<a href="https://pypi.org/project/trapper"><img src="https://img.shields.io/pypi/pyversions/trapper" alt="Python versions"></a>
<a href="https://pypi.org/project/trapper"><img src="https://img.shields.io/pypi/v/trapper?color=blue" alt="PyPI version"></a>
<a href="https://github.com/obss/trapper/releases/latest"><img alt="Latest Release" src="https://img.shields.io/github/release-date/obss/trapper"></a>
<a href="https://colab.research.google.com/github/obss/trapper/blob/main/examples/question_answering/question_answering.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
<br>
<a href="https://github.com/obss/trapper/actions"><img alt="Build status" src="https://github.com/obss/trapper/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://libraries.io/pypi/trapper"><img alt="Dependencies" src="https://img.shields.io/librariesio/release/pypi/trapper"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/obss/trapper/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/obss/trapper"></a>
</p>

Trapper is an NLP library that aims to make it easier to train transformer based
models on downstream tasks. It
wraps [huggingface/transformers](http://github.com/huggingface/transformers) to
provide the transformer model implementations and training mechanisms. It defines
abstractions with base classes for common tasks encountered while using transformer
models. Additionally, it provides a dependency-injection mechanism and allows
defining training and/or evaluation experiments via configuration files. By this
way, you can replicate your experiment with different models, optimizers etc by only
changing their values inside the configuration file without writing any new code or
changing the existing code. These features foster code reuse, less boiler-plate
code, as well as repeatable and better documented training experiments which is
crucial in machine learning.

## Why You Should Use Trapper

- You have been a `Transformers` user for quite some time now. However, you started
  to feel that some computation steps could be standardized through new
  abstractions. You wish to reuse the scripts you write for data processing,
  post-processing etc with different models/tokenizers easily. You would like to
  separate the code from the experiment details, mix and match components through
  configuration files while keeping your codebase clean and free of duplication.


- You are an `AllenNLP` user who is really happy with the dependency-injection
  system, well-defined abstractions and smooth workflow. However, you would like to
  use the latest transformer models without having to wait for the core developers
  to integrate them. Moreover, the `Transformers` community is scaling up rapidly,
  and you would like to join the party while still enjoying an `AllenNLP` touch.


- You are an NLP researcher / practitioner, and you would like to give a shot to a
  library aiming to support state-of-the-art models along with datasets, metrics and
  more in unified APIs.

To see more, check the 
[official Trapper blog post](https://medium.com/codable/trapper-an-nlp-library-for-transformer-models-b8917bbc8796).

## Key Features

### Compatibility with HuggingFace Transformers

**Trapper extends Transformers!**

While implementing the components of trapper, we try to reuse the classes from the
Transformers library as much as we can. For example, trapper uses the models, and
the trainer as they are in Transformers. This makes it easy to use the models
trained with trapper on other projects or libraries that depend on Transformers
(or pytorch in general).

We strive to keep trapper fully compatible with Transformers, so you can always use
some of our components to write a script for your own needs while not using the full
pipeline (e.g. for training).

### Dependency Injection and Training Based on Configuration Files

We use the registry mechanism of [AllenNLP](http://github.com/allenai/allennlp) to
provide dependency injection and enable reading the experiment details from the
configuration files in `json`
or `jsonnet` format. You can look at the
[AllenNLP guide on dependency injection](https://guide.allennlp.org/using-config-files)
to learn more about how the registry system and dependency injection works as well
as how to write configuration files. In addition, we strongly recommend reading the
remaining parts of the [AllenNLP guide](https://guide.allennlp.org/)
to learn more about its design philosophy, the importance of abstractions etc.
(especially Part2: Abstraction, Design and Testing). As a warning, please note that
we do not use AllenNLP's abstractions and base classes in general, which means you
can not mix and match the trapper's and AllenNLP's components. Instead, we just use
the class registry and dependency injection mechanisms and only adapt its very
limited set of components, first by wrapping and registering them as trapper
components. For example, we use the optimizers from AllenNLP since we can
conveniently do so without hindering our full compatibility with Transformers.

### Full Integration with HuggingFace Datasets

In trapper, we officially use the format of the datasets
from [datasets](http://github.com/huggingface/datasets) and provide full integration
with it. You can directly use all datasets published
in [datasets hub](https://huggingface.co/datasets) without doing any extra work. You
can write the dataset name and extra loading arguments (if there are any) in your
training config file, and trapper will automatically download the dataset and pass
it to the trainer. If you have a local or private dataset, you can still use it
after converting it to the HuggingFace `datasets` format by writing a dataset
loading script as explained
[here](https://huggingface.co/docs/datasets/dataset_script.html).

### Support for Metrics through Jury

Trapper supports the common NLP metrics through
[jury](https://github.com/obss/jury). Jury is an NLP library dedicated to provide
metric implementations by adopting and extending the datasets library. For metric
computation during training you can use jury style metric
instantiation/configuration to set up on your trapper configuration file to compute
metrics on the fly on eval dataset with a specified `eval_steps` value. If your
desired metric is not yet available on jury or datasets, you can still create your
own by extending `trapper.Metric` and utilizing either
`jury.Metric` or `datasets.Metric` for handling larger set of cases on predictions.

### Abstractions and Base Classes

Following AllenNLP, we implement our own registrable base classes to abstract away
the common operations for data processing and model training.

* Data reading and preprocessing base classes including

    - The classes to be used directly: `DatasetReader`, `DatasetLoader`
      and `DataCollator`.

    - The classes that you may need to extend: `LabelMapper`,`DataProcessor`
      , `DataAdapter` and `TokenizerWrapper`.

    - `TokenizerWrapper` classes utilizing `AutoTokenizer` from Transformers are
      used as factories to instantiate wrapped tokenizers into which task-specific
      special tokens are registered automatically.


* `ModelWrapper` classes utilizing the `AutoModelFor...` classes from Transformers
  are used as factories to instantiate the actual task-specific models from the
  configuration files dynamically.

* Optimizers from AllenNLP: Implemented as children of the base `Optimizer` class.

* Metric computation is supported through `jury`. In order to make the metrics
  flexible enough to work with the trainer in a common interface, we introduced
  metric handlers. You may need to extend these classes accordingly
    * For conversion of predictions and references to a suitable form for a
      particular metric or metric set: `MetricInputHandler`.
    * For manipulating resulting score object containing the metric
      results: `MetricOutputHandler`.

## Usage

To use trapper, you need to select the common NLP formulation of the problem you are
tackling as well as decide on its input representation, including the special
tokens.

### Modeling the Problem

The first step in using trapper is to decide on how to model the problem. First, you
need to model your problem as one of the common modeling tasks in NLP such as
seq-to-seq, sequence classification etc. We stick with the Transformers' way of
dividing the tasks into common categories as it does in its `AutoModelFor...`
classes. To be compatible with Transformers and reuse its model factories, trapper
formalizes the tasks by wrapping the `AutoModelFor...` classes and matching them to
a name that represents a common task in NLP. For example, the natural choice for POS
tagging is to model it as a token classification (i.e. sequence labeling) task. On
the other hand, for question answering task, you can directly use the question
answering formulation since Transformers already has a support for that task.

### Modeling the Input

You need to decide on how to represent the input including the common special tokens
such as BOS, EOS. This formulation is directly used while creating the
`input_ids` value of the input instances. As a concrete example, you can represent a
sequence classification input with `BOS ... actual_input_tokens ... EOS` format.
Moreover, some tasks require extra task-specific special tokens as well. For
example, in conditional text generation, you may need to prompt the generation with
a special signaling token. In tasks that utilizes multiple sequences, you may need
to use segment embeddings (via `token_type_ids`) to label the tokens according to
their sequence.

## Class Reference

<p align="center">
  <img src="https://github.com/obss/trapper/blob/main/resources/trapper_diagram.png?raw=true" alt="trapper_components"/>
</p>

The above diagram shows the basic components in Trapper. To use trapper on training,
evaluation on a task that is not readily supported in Transformers, you need to
extend the provided base classes according to your own needs. These are as follows:

**For Training & Evaluation**: LabelMapper, DataProcessor, DataAdapter,
TokenizerWrapper, MetricInputHandler, MetricOutputHandler.

**For Inference**: In addition to the ones listed above, you may need to implement
a `transformers.Pipeline` or directly use one from Transformers if they already
implemented one that matches your need.

**Typically Extended Classes**

1) **LabelMapper**:
   Used in tasks that require mapping between categorical labels and integer ids
   such as token classification.


2) **DataProcessor**:
   This class is responsible for taking a single instance in dict format, typically
   coming from a `datasets.Dataset`, extracting the information fields suitable for
   the task and hand, and converting their values to integers or collections of
   integers. This includes, tokenizing the string fields, and getting the token ids,
   converting the categoric labels to integer ids and so on.


3) **DataAdapter**:
   This is responsible for converting the information fields inside an instance dict
   that was previously processed by a `DataProcessor` to a format suitable for
   feeding into a transformer model. This also includes handling the special tokens
   signaling the start or end of a sequence, the separation of tho sequence for a
   sequence-pair task as well as chopping excess tokens etc.


4) **TokenizerWrapper**:
   This class wraps a pretrained tokenizer from the Transformers while also
   recording the special tokens needed for the task to the wrapped tokenizer. It
   also stores the missing values from BOS - CLS, EOS - SEP token pairs for the
   tokenizers that only support one of them. This means you can model your input
   sequence by using the bos_token for start and eos_token for end without thinking
   which model you are working with. If your task and input modeling needs extra
   special tokens e.g. the `<CONTEXT>` for a context dependent task, you can store
   these tokens by setting the `_TASK_SPECIFIC_SPECIAL_TOKENS` class variable in
   your TokenizerWrapper subclass. Otherwise, you can directly use TokenizerWrapper.


5) **MetricInputHandler**:
   This class is mainly responsible for preprocessing applied to predictions and
   labels (references). This is performed for transforming the predictions and
   labels to a suitable format to be fed in metrics for computation. For example,
   while using BLEU in a language generation task, the predictions and labels need
   to be converted to a string or list of strings. However, for extractive question
   answering task in which the predictions are returned as start and end indices
   pointing the answer within the context, additional information (e.g context in
   such case) may be needed, so directly returning the start and end indices in this
   case does not help, and additional operation is needed to be done by converting
   predictions to actual answers extracted from the context. You are able to do this
   kind of operations through `MetricInputHandler`, storing additional information,
   converting predictions and labels to a suitable format, manipulating resulting
   score. Furthermore, in child classes helper classes can also be implemented (e.g
   `TokenizerWrapper`, `LabelMapper`) for required tasks. In this class, we provide
   three main functionality:
    * `_extract_metadata()`: This method allows user to extract metadata from
      dataset instances to be later used for preprocessing predictions and labels
      in `preprocess()` method.
    * `__call__()`: This method allows converting predictions and labels into a
      suitable form for metric computation. The default behavior is defined as
      directly returning predictions and labels without manipulation, but only
      applying `argmax()` to predictions to convert the model predictions to
      predictions input for metrics.

7) **MetricOutputHandler**:
   The intention of this class is to support for manipulating the score object
   returned by the metric computation phase. Jury returns a well-constructed
   dictionary output for all metrics; however, to shorten dictionary items,
   manipulate the information within the output or to add additional information to
   score dictionary, this class can be extended as desired.


7) **transformers.Pipeline**:
   The pipeline mechanism from Transformers have not been fully integrated yet. For
   now, you should check Transformers to find a pipeline that is suitable for your
   needs and does the same pre-processing. If you could not find one, you may need
   to write your own `Pipeline` by extending
   `transformers.Pipeline` or one of its subclasses and add it
   to `transformers.pipelines.SUPPORTED_TASKS` map. To enable instantiation of the
   pipelines from the checkpoint folders, we provide a factory,
   `create_pipeline_from_checkpoint` function. It accepts a checkpoint directory of
   a completed experiment, the path to the config file (already saved in that
   directory), as well as the task name that you used while adding the pipeline
   to `SUPPORTED_TASKS`. It re-creates the objects you used while training such
   as `model wrapper`, `label mapper` etc and provides them as keyword arguments to
   constructor of the pipeline you implemented.

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

##### Script Based Training

Go to your project root and execute the `trapper run` command with a config file
specifying the details of the training experiment. E.g.

```shell
trapper run SOME_DIRECTORY/experiment.jsonnet
```

Don't forget to provide the args["output_dir"] and args["result_dir"] values in your
experiment file. Please look at the `examples/pos_tagging/README.md` for a detailed
example.

## Examples for Using Trapper as a Library

We created an `examples` folder that includes example projects to help you get
started using trapper. Currently, it includes a POS tagging project using the
CONLL2003 dataset, and a question answering project using the SQuAD dataset. The POS
tagging example shows how to use trapper on a task that does not have a direct
support from Transformers. It implements all custom components and provides a
complete project structure including the tests. On the other hand, the question
answering example shows using trapper on a task that Transformers already supported.
We implemented it to demonstrate how trapper may still be helpful thanks to
configuration file based experiments.

### Training a POS Tagging Model on CONLL2003

Since transformers lacks a direct support for POS tagging, we added an
[example project](./examples/pos_tagging) that trains a transformer model
on `CONLL2003` POS tagging dataset and perform inference using it. It is a
self-contained project including its own requirements file, therefore you can copy
the folder into another directory to use as a template for your own project. Please
follow its `README.md` to get started.

### Training a Question Answering Model on SQuAD Dataset

You can use the notebook in
the [Example QA Project](./examples/question_answering) `examples/question_answering/question_answering.ipynb`
to follow the steps while training a transformer model on SQuAD v1.

## Currently Supported Tasks and Models From Transformers

Hypothetically, nearly all models should work on any task if it has an entry in the
table of `AutoModelFor...` factories for that task. However, since some models
require more (or less) parameters compared to most of the models in the library, you
might get errors while using such models. We try to cover these edge cases them by
adding the extra parameters they require. Feel free to open an issue/PR if you
encounter/solve such issues in a model-task combination. We have used trapper on a
limited set of model-task combinations so far. We list these combinations below to
indicate that they have been tested and validated to work without problems.

### Table of Model-task Combinations Tested so far

| model       | question_answering | token_classification |
|-------------|--------------------|----------------------|
| BERT        | &#10004;           | &#10004;             |
| ALBERT      | &#10004;           | &#10004;             |
| DistillBERT | &#10004;           | &#10004;             |
| ELECTRA     | &#10004;           | &#10004;             |
| RoBERTa     | &#10004;           | &#10004;             |

## Installation

### Environment Creation

It is strongly recommended creating a virtual environment using conda or virtualenv
etc. before installing this package and its dependencies. For example, the following
code creates a conda environment with name trapper and python version 3.7.10, and
activates it.

```console
conda create --name trapper python=3.7.10
conda activate trapper
```

#### Regular Installation

You can install trapper and its dependencies by pip as follows.

```console
pip install trapper
```

## Contributing

If you would like to open a PR, please create a fresh environment as described
before, clone the repo locally and install trapper in editable mode as follows.

```console
git clone https://github.com/obss/trapper.git
cd trapper
pip install -e .[dev]
```

After your changes, please ensure that the tests are still passing, and do not
forget to apply code style formatting.

### Testing trapper

#### Caching the test fixtures from the HuggingFace datasets library

To speed up the data-related tests, we cache the test dataset fixtures from
HuggingFace's datasets library using the following command.

```console
python -m scripts.cache_hf_datasets_fixtures
```

Then, you can simply run the tests with the following command:

```console
python -m scripts.run_tests
```

**NOTE:** To significantly speed up the tests depending on HuggingFace's
Transformers and datasets libraries, you can set the following environment variables
to make them work in offline mode. However, beware that you may need to run the
tests once first without setting these environment variables so that the pretrained
models, tokenizers etc. are downloaded and cached.

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

### Contributors

- [Cemil Cengiz](https://github.com/cemilcengiz)
- [Devrim Çavuşoğlu](https://github.com/devrimcavusoglu)