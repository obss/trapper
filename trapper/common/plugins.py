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
# Plugin discovery and import functions.

trapper supports registering classes from custom modules/packages etc written by the
user. Parts of this file is adapted from the AllenNLP library at
https://github.com/allenai/allennlp.
"""

import importlib
import os
import sys
from typing import Iterable, Set

from allennlp.common.plugins import discover_file_plugins
from allennlp.common.util import push_python_path
from transformers.utils import logging

logger = logging.get_logger(__name__)

LOCAL_PLUGINS_FILENAME = ".trapper_plugins"
"""
Local plugin files should have this name.
"""


def discover_plugins() -> Iterable[str]:
    """
    Returns an iterable of the plugins found in the local plugin file.
    """
    plugins: Set[str] = set()
    if os.path.isfile(LOCAL_PLUGINS_FILENAME):
        with push_python_path("."):
            for plugin in discover_file_plugins(LOCAL_PLUGINS_FILENAME):
                if plugin in plugins:
                    continue
                yield plugin
                plugins.add(plugin)


def import_plugins() -> None:
    """
    Imports the plugins found with `discover_plugins()` i.e. the custom
    registrable components written by the user.
    """
    # For a presumed Python issue that makes the spawned processes unable
    # to find modules in the current directory.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    for module_name in discover_plugins():
        try:
            importlib.import_module(module_name)
            logger.info("Plugin %s available", module_name)
        except ModuleNotFoundError as e:
            logger.error(f"Plugin {module_name} could not be loaded: {e}")
