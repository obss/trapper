# Copyright 2021 Open Business Software Solutions, the AllenNLP library authors. All rights reserved.
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
Commands to interact with trapper through the CLI. Parts of this file is adapted
from the AllenNLP library at https://github.com/allenai/allennlp.
"""

import argparse
import json
import sys
import uuid
from abc import ABCMeta
from typing import Dict, List, Optional, Set, Tuple, Type

import allennlp as _allennlp
from allennlp.commands import ArgumentParserWithDefaults
from allennlp.common.util import import_module_and_submodules
from jury.utils.common import replace as list_replace
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import elastic_launch
from torch.distributed.run import config_from_args as torch_config_from_args
from torch.distributed.run import get_args_parser as torch_distributed_args_parser

from trapper import PROJECT_ROOT, __version__
from trapper.common.plugins import import_plugins
from trapper.common.utils import append_parent_docstr
from trapper.training.train import run_experiment


@append_parent_docstr
class Subcommand(_allennlp.commands.Subcommand, metaclass=ABCMeta):
    """
    This class is created to get the command mechanism of the allennlp library.
    """

    requires_plugins: bool = True
    _reverse_registry: Dict[Type, str] = {}


# Getting rid of unused commands in allennlp
_allennlp.commands.Subcommand = Subcommand

log = get_logger()


@Subcommand.register("run")
class Run(Subcommand):
    """trapper's main command that enables creating and running an experiment
    form a config file.

    Usage:
        Basic:
            ` trapper run --config_path experiment.jsonnet `

        With overrides flag:
           ` trapper run --config_path experiment.jsonnet `
    """

    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Run the experiment specified in the config file which
         specify the training and/or evaluation of a model on a dataset."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Train and/or evaluate a model.",
        )

        subparser.add_argument(
            "config_path",
            type=str,
            help="path to the experiment config file in `json` or `jsonnet`"
            " format describing the model, dataset and other details.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json or jsonnet structure used to override the experiment "
                "configuration, e.g., '{\"optimizer.lr\": 1e-5}'.  Nested "
                "parameters can be specified either with nested dictionaries "
                "or with dot syntax."
            ),
        )

        subparser.set_defaults(func=run_experiment_from_args)

        return subparser


@Subcommand.register("run-distributed")
class DistributedRun(Subcommand):
    """Start an experiment in a distributed data parallel fashion using the config file.
    It utilizes pytorch's elastic run under the hood. To see supported features see
    https://pytorch.org/docs/stable/elastic/run.html

    Usage:
        Basic:
            ` trapper run-distributed experiment.jsonnet <--other-kwargs vals>`
    """

    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Run the experiment specified in the config file in a 
            distributed fashion across multiple devices (DDP)."""
        subparser = parser.add_parser(
            self.name,
            description=description,
            help="Start distributed training of a model.",
        )

        subparser.add_argument(
            "config_path",
            type=str,
            help="path to the experiment config file in `json` or `jsonnet`"
            " format describing the model, dataset and other details.",
        )

        subparser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help=(
                "a json or jsonnet structure used to override the experiment "
                "configuration, e.g., '{\"optimizer.lr\": 1e-5}'.  Nested "
                "parameters can be specified either with nested dictionaries "
                "or with dot syntax."
            ),
        )

        subparser.set_defaults(func=run_distributed)

        return subparser


@record
def run_distributed(args=None):
    """
    Taken from
    https://github.com/pytorch/pytorch/blob/4618371da56c887195e2e1d16dad2b9686302800/torch/distributed/run.py
    and modified to properly run by trapper.
    """

    def merge_json_str(opts: List[str]) -> None:
        """
        Merges the JSON structures from given command line arguments to correct
        the JSON structure passed by `--overrides` argument. Implements a
        try/except block to merge json. Note that it modifies given args
        list inplace.

        Notes:
            The argparser may divide the given json structure to multiple
            shards if it has whitespace between key and value (e.g.
            {\"key\": \"abc\"} -argparse-> ['{\"key\":', '\"abc\"}']). This
            function takes the sharded arguments and merges them if they
            construct a valid JSON structure.

        Args:
            opts:

        Returns:
            List of strings json where JSON structures merged.
        """
        start = step = 1

        i = 0
        while i < len(opts):
            if opts[i].count("{") >= 1:
                start = i
                break
            i += 1

        while start + step < len(opts) + 1:
            end = start + step
            try:
                json_shards = " ".join(opts[start:end])
                json.loads(json_shards)
            except json.decoder.JSONDecodeError:
                step += 1
            else:
                json_str = " ".join(opts[start:end])
                del opts[start:end]
                opts.insert(start, json_str)
                break

    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:29400"
        args.rdzv_id = str(uuid.uuid4())
        log.info(
            f"\n**************************************\n"
            f"Rendezvous info:\n"
            f"--rdzv_backend={args.rdzv_backend} "
            f"--rdzv_endpoint={args.rdzv_endpoint} "
            f"--rdzv_id={args.rdzv_id}\n"
            f"**************************************\n"
        )

    config, cmd, cmd_args = torch_config_from_args(args)

    # Monkey-patching the original args extracted from torch's
    # config parser. Effectively, running the following cmd
    # $ torchrun TORCHRUN_OPTS /path/to/trapper/__main__.py run TRAPPER_OPTS
    #
    merge_json_str(cmd_args)
    training_script_idx = cmd_args.index("run-distributed")
    trapper_main_script = str(PROJECT_ROOT / "trapper/__main__.py")
    list_replace(cmd_args, trapper_main_script, training_script_idx)
    cmd_args.insert(training_script_idx + 1, "run")

    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


def torch_distributed_parse_args():
    torch_parser = torch_distributed_args_parser()
    dist_args, _ = torch_parser.parse_known_args()
    for flag in dist_args.training_script_args[1:]:
        if "=" in flag:
            # torch parser can only parse args if '=' is used between key and value
            # we do not perform any polishing merge of keys and values
            opt, f, v = torch_parser._parse_optional(flag)
            if opt is not None:
                setattr(dist_args, opt.dest, v)
                dist_args.training_script_args.remove(flag)
    return dist_args


def run_experiment_from_args(args: argparse.Namespace):
    """
    Unpacks the `argparse.Namespace` object to initiate an experiment.
    """
    run_experiment(args.config_path, args.overrides)


def merge_args_safe(
    args1: argparse.Namespace, args2: argparse.Namespace
) -> argparse.Namespace:
    """
    Merges two namespaces but throws an error if there are keys that collide.

    ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
    :param args1:
    :param args2:
    :return:
    """
    # - the merged args
    # The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
    merged_args = argparse.Namespace(**vars(args1), **vars(args2))
    return merged_args


def parse_args(
    prog: Optional[str] = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Creates the argument parser for the main program and uses it to parse the args.
    (Note: This function is adapted from `allennlp.commands.__init__.parse_args`).
    """

    parser = ArgumentParserWithDefaults(description="Run Trapper", prog=prog)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands: Set[str] = set()

    def add_subcommands():
        for subcommand_name in sorted(Subcommand.list_available()):
            if subcommand_name in subcommands:
                continue
            subcommands.add(subcommand_name)
            subcommand_class = Subcommand.by_name(subcommand_name)
            subcommand = subcommand_class()
            subparser = subcommand.add_subparser(subparsers)
            if subcommand_class.requires_plugins:  # type: ignore
                subparser.add_argument(
                    "--include-package",
                    type=str,
                    action="append",
                    default=[],
                    help="additional packages to include",
                )

    # Add all default registered subcommands first.
    add_subcommands()

    # If we need to print the usage/help, or the subcommand is unknown,
    # we'll call `import_plugins()` to register any plugin subcommands first.
    argv = sys.argv[1:]
    plugins_imported: bool = False
    if not argv or argv == ["--help"] or argv[0] not in subcommands:
        import_plugins()
        plugins_imported = True
        # Add subcommands again in case one of the plugins has a registered subcommand.
        add_subcommands()

    # Now we can parse the arguments.
    args, _ = parser.parse_known_args()
    if argv[0] == "run-distributed":
        dist_args = torch_distributed_parse_args()
        args = merge_args_safe(args, dist_args)

    if (
        not plugins_imported and Subcommand.by_name(argv[0]).requires_plugins
    ):  # type: ignore
        import_plugins()

    return parser, args


def main(prog: Optional[str] = None) -> None:
    """
    The [`run`] command only knows about the registered classes in ``trapper``
     codebase. In particular, it won't work for your own models, dataset readers
     etc unless you use the ``--include-package`` flag or you make your code
     available as a plugin. (Note: This function is copied from
     `allennlp.commands.__init__.main`.)
    """
    parser, args = parse_args(prog)

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, "include_package", []):
            import_module_and_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main("trapper")
