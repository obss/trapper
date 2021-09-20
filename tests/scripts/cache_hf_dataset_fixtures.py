from tests.scripts.utils import shell, validate_and_exit

if __name__ == "__main__":
    validate_and_exit(
        shell(
            "datasets-cli test test_data/hf_datasets/squad_qa_test_fixture --save_infos --all_configs"
        )
    )
