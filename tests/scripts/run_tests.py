from tests.scripts.utils import shell, validate_and_exit

if __name__ == "__main__":
    sts_tests = shell(
        "pytest --cov trapper --cov-report term-missing --cov-report xml -vvv tests"
    )
    validate_and_exit(tests=sts_tests)
