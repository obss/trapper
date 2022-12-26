from trapper.common.testing_utils.shell_utils import shell, validate_and_exit

if __name__ == "__main__":
    sts_tests = shell(
        "pytest --cov trapper --cov-report term-missing --cov-report xml -vvv tests"
    )
    sts_tests_examples = shell(
        "cd examples/pos_tagging && python -m scripts.run_tests"
    )
    validate_and_exit(tests=sts_tests, tests_examples=sts_tests_examples)
