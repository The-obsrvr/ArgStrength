from __future__ import absolute_import, division, print_function

# Local Imports
from modeling import ArgStrModel
from arguments import TrainingArguments
from training import ArgStrTrainer
from processors import get_datasets
from utils import Data_Collator, set_seed

# Standard Imports
import os
import random

# Third Party Imports
import json
import torch
import mlflow

from transformers import BertConfig

if __name__ == '__main__':
    # load the best trained model and config
    config_available = False

    # define the experiment parameters
    bert_arch = "bert-base-uncased"
    task_name = "STLAS_only_toledo_randomized"
    exp_name = task_name + "_v2_bb"
    mlflow_exp_name = "ASL_randomized"

    # define the path from where the task data is loaded from.
    task_data_dir = "/mnt/data2/Sid/arg_quality/pytorch/task4_hpo/data/*.csv"

    if config_available:
        config_file = open("/mnt/data2/Sid/arg_quality/pytorch/task4_hpo/best_model_details_MTLAS_LOO_swanson_v2_bb"
                           ".json", )
        config_data = json.load(config_file)
    else:
        # define config
        config_data = {
            "bert_arch": bert_arch,
            "task_name": task_name,
            "exp_name": exp_name,
            "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            "is_distributed": False,
            "resources_per_trial": {
                'cpu': 4,
                'gpu': 1
            },

            "data_dir": task_data_dir,
            "split_by_topic": True if "randomized" in task_name else False,
            "eval_batch_size": 128,
            "train_batch_size": 64,
            "max_seq_length": None,
            "max_seq_length_perc": 0.95,
            "data_collator": Data_Collator,

            "dropout_prob": 0.1,
            "bert_hidden_layers": 4,
            "nn_config": 1,
            "dataset_loss_method": "weighted",

            "learning_rate": 4.8672041684500765e-06,
            "weight_decay": 0.1936871758204528,
            "num_epochs": 20,
            "max_steps": -1,  # We use num_epochs instead.

            "mlflow": {
                "experiment_name": mlflow_exp_name,
                "tracking_uri": "http://mlflow.dbs.ifi.lmu.de:5000"
            }
        }

    if mlflow.get_experiment_by_name(mlflow_exp_name) is None:
        mlflow.create_experiment(mlflow_exp_name)
    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    mlflow.set_experiment(experiment_name=mlflow_exp_name)
    # define 10 seeds to run for training
    seed_list = random.sample(range(0, 10000), 10)
    for seed_value in seed_list:
        set_seed(seed_value)
        with mlflow.start_run():
            mlflow.log_param("seed", seed_value)
            mlflow.log_param("Task Name", task_name)
            print("Seed:", seed_value)
            train_dataset, eval_dataset, test_dataset, task_dict = get_datasets(config_data)
            # Load model setup.
            if not config_available:
                bert_config = BertConfig.from_pretrained(
                    config_data["bert_arch"],
                    finetuning_task=task_name,
                    output_hidden_states=True
                )

                model = ArgStrModel.from_pretrained(
                    config_data["bert_arch"],
                    config=bert_config,
                    dropout_prob=config_data["dropout_prob"],
                    bert_hidden_layers=config_data["bert_hidden_layers"],
                    mlp_config=config_data["nn_config"],
                    task_dict=task_dict,
                    device=config_data["device"]
                )
            else:
                bert_config = BertConfig.from_pretrained(
                    config_data["bert_arch"],
                    finetuning_task=task_name,
                    output_hidden_states=True
                )
                model = ArgStrModel(
                    config=bert_config,
                    dropout_prob=config_data["dropout_prob"],
                    bert_hidden_layers=config_data["bert_hidden_layers"],
                    mlp_config=config_data["nn_config"],
                    task_dict=task_dict,
                    device=config_data["device"])

                with open(os.path.join(config_data["best_checkpoint_path"], "best_model.pt"), 'rb') as checkpt:
                    model_state, optimizer_state = torch.load(checkpt)
                model.load_state_dict(model_state)

            training_args = TrainingArguments(
                learning_rate=config_data["learning_rate"],
                train_model=True,
                evaluate_during_training=True,
                save_steps=0,
                max_num_train_epochs=20,
                train_batch_size=config_data["train_batch_size"],
                eval_batch_size=config_data["eval_batch_size"],
                weight_decay=config_data["weight_decay"],
                weighted_dataset_loss=config_data["dataset_loss_method"]
            )

            retrain_runner = ArgStrTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                test_dataset=test_dataset,
                data_collator=Data_Collator,
                task_name=config_data["task_name"]
            )

            retrain_runner.train_model(device=config_data["device"],
                                       mlflow_logging=True,
                                       retraining=True,
                                       seed_value=seed_value)

        mlflow.end_run()