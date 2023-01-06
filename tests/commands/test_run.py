import pytest
import sys
from unittest.mock import patch
from trapper.commands import main


def test_run_command_missing(capsys):
    with pytest.raises(SystemExit, match='2'):
        run_args = ["trapper", "run"]
        with patch.object(sys, 'argv', run_args):
            main('trapper')

    captured = capsys.readouterr()
    assert "trapper run: error: the following arguments are required" in captured.err