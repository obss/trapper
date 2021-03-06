from trapper.common.testing_utils.shell_utils import shell, validate_and_exit

if __name__ == "__main__":
    sts_tests = shell(
        "pytest --cov src --cov-report term-missing --cov-report xml -vvv tests"
    )
    validate_and_exit(tests=sts_tests)
