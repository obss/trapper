import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
)


def run():
    from trapper.commands import main  # noqa

    main(prog="trapper")


if __name__ == "__main__":
    run()
