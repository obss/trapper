
### Installation

#### Environment Creation

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