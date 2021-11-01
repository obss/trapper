local output_dir = "experiments/roberta/outputs";
local result_dir = "experiments/roberta/results";
local conll2003_test_fixture="test_fixtures/hf_datasets/conll2003_test_fixture";
local save_steps = 292;
{
        "pretrained_model_name_or_path": "roberta-base",
        "train_split_name": "train",
        "dev_split_name": "validation",
        "tokenizer_wrapper": {
            "type": "pos_tagging_example",
            "model_max_sequence_length": 512,
            "add_prefix_space": true
        },
        "dataset_loader": {
            "dataset_reader": {
                "path": "conll2003",  # actual dataset
//                "path": conll2003_test_fixture,  # for testing the project
            },
            "data_processor": {"type": "conll2003_pos_tagging_example"},
            "data_adapter": {"type": "conll2003_pos_tagging_example"},
        },
        "data_collator": {"type": "default"},
        "model_wrapper": {"type": "token_classification", "num_labels": 47},
        "compute_metrics": {"type": "seqeval",
                            "return_entity_level_metrics": false},
        "label_mapper": {"type": "conll2003_pos_tagging_example"},
        "args": {
            "type": "default",
            "output_dir": output_dir + "/checkpoints",
            "result_dir": result_dir,
            "logging_first_step": true,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 32,
            "gradient_accumulation_steps": 1,
            "logging_dir": output_dir + "/logs",
            "no_cuda": true,
            "logging_steps": save_steps,
            "eval_steps": save_steps,
            "evaluation_strategy": "steps",
            "save_steps": save_steps,
            "label_names": ["labels"],
            "lr_scheduler_type": "linear",
            "warmup_steps": 157,
            "do_train": true,
            "do_eval": true,
            "save_total_limit": 1,
            "metric_for_best_model": "eval_f1",
            "greater_is_better": true,
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "weight_decay": 0.1,
            "parameter_groups": [
                [
                    ["bias", "LayerNorm\\\\.weight", "layer_norm\\\\.weight"],
                    {"weight_decay": 0},
                ]
            ],
            "lr": 3e-5,
            "eps": 1e-8,
        },
    }
