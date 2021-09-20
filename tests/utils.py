import os
import sys


def shell(command, exit_status=0):
    """
    Run command through shell and return exit status if exit status of command run match with given exit status.

    Args:
        command: (str) Command string which runs through system shell.
        exit_status: (int) Expected exit status of given command run.

    Returns: actual_exit_status

    """
    actual_exit_status = os.system(command)
    if actual_exit_status == exit_status:
        return 0
    return actual_exit_status


def validate_and_exit(*args, expected_out_status=0):
    if all([arg == expected_out_status for arg in args]):
        sys.exit(0)
    else:
        sys.exit(256)
