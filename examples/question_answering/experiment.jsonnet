local checkpoint_dir = std.extVar("CHECKPOINT_PATH");
local result_dir = std.extVar("OUTPUT_PATH");
{
    "train_split_name": "train",
    "dev_split_name": "validation",
    "pretrained_model_name_or_path": "roberta-base",
    "tokenizer_wrapper": {
        "type": "question-answering"
    },
    "dataset_loader": {
        "type": "default",
        "dataset_reader": {
            "type": "default",
            "path": "../../test_fixtures/hf_datasets/squad_qa_test_fixture"
        },
        "data_processor": {
            "type": "squad-question-answering"
        },
        "data_adapter": {
            "type": "question-answering"
        },
        "metadata_handler": {
            "type": "question-answering"
        }
    },
    "data_collator":{
        "type": "default"
    },
    "model_wrapper": {
        "type": "question_answering"
    },
    "compute_metrics": {
        "metric_params": [
            "squad"
        ]
    },
    "args": {
        "type": "default",
        "output_dir": checkpoint_dir,
        "result_dir": result_dir,
        "num_train_epochs": 10,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 12,
        "per_device_eval_batch_size": 2,
        "logging_dir": checkpoint_dir + "/logs",
        "no_cuda": false,
        "logging_steps": 500,
        "evaluation_strategy": "steps",
        "save_steps": 500,
        "label_names": ["start_positions", "end_positions"],
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "do_train": true,
        "do_eval": true,
        "save_total_limit": 1
    },
    "optimizer": {
        "type": "huggingface_adamw",
        "weight_decay": 0.01,
        "parameter_groups": [
            [["bias", "LayerNorm\\\\.weight", "layer_norm\\\\.weight"],
             {"weight_decay": 0}]],
        "lr": 5e-5,
        "eps": 1e-6
    }
}