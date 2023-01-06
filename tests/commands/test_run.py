import pytest
import sys
from unittest.mock import patch
from trapper.commands import main
from trapper import FIXTURES_ROOT

COMMAND_FIXTURES = FIXTURES_ROOT / "commands"

def test_run_command_missing(capsys):
    with pytest.raises(SystemExit, match='2'):
        run_args = ["trapper", "run"]
        with patch.object(sys, 'argv', run_args):
            main('trapper')

    captured = capsys.readouterr()
    assert "trapper run: error: the following arguments are required" in captured.err


def test_run_command_trivial(tmp_path):
    run_args = ["trapper", "run", str(COMMAND_FIXTURES / "experiment_trivial.jsonnet"), "-o"]
    overrides = {
        "args.output_dir": str(tmp_path / "output"),
        "args.result_dir": str(tmp_path / "output"),
    }
    run_args.append(str(overrides))
    with patch.object(sys, 'argv', run_args):
        main('trapper')
