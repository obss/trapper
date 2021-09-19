from tests.scripts.utils import assert_shell

if __name__ == "__main__":
    assert_shell(
        "datasets-cli test test_data/hf_datasets/squad_qa_test_fixture --save_infos --all_configs"
    )
