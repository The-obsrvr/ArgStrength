# Local Imports
from modeling import ArgStrModel
from training import ArgStrTrainer
from utils import Data_Collator, load_data_files, TestResult, RegressionResult, recover_checkpoint
from arguments import TrainingArguments
from processors import get_datasets

# Standard Imports
import os
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Third-Party Imports
import json
import mlflow
import ray
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter
import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import BertConfig


# Create MLflow experiment
# mlflow.create_experiment("sid_task4_hpov1")


@mlflow_mixin
def train_bert(config, checkpoint_dir=None):
    """

    :param config:
    :param checkpoint_dir:
    :return:
    """

    train_dataset, eval_dataset, test_dataset, task_dict = get_datasets(config)

    training_args = TrainingArguments(
        output_dir=tune.get_trial_dir(),
        learning_rate=config["learning_rate"],
        train_model=True,
        evaluate_during_training=True,
        # Run eval after every epoch.
        eval_steps=(len(train_dataset) // config["train_batch_size"]) + 1,
        # We explicitly set save to 0, and do checkpointing in evaluate instead
        save_steps=0,
        max_num_train_epochs=config["num_epochs"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        weight_decay=config["weight_decay"],
        weighted_dataset_loss=config["dataset_loss_method"],
        is_distributed=config["is_distributed"],
    )
    optimizer_state = None
    model_name_or_path = recover_checkpoint(checkpoint_dir, config["bert_arch"])

    if model_name_or_path == config["bert_arch"]:
        bert_config = BertConfig.from_pretrained(
            model_name_or_path,
            finetuning_task=task_name,
            output_hidden_states=True
        )

        model = ArgStrModel.from_pretrained(
            model_name_or_path,
            config=bert_config,
            dropout_prob=config["dropout_prob"],
            bert_hidden_layers=config["bert_hidden_layers"],
            mlp_config=config["nn_config"],
            task_dict=task_dict,
            device=config["device"]
        )
    else:
        bert_config = BertConfig.from_pretrained(
            config["bert_arch"],
            finetuning_task=task_name,
            output_hidden_states=True
        )
        model = ArgStrModel(
            config=bert_config,
            dropout_prob=config["dropout_prob"],
            bert_hidden_layers=config["bert_hidden_layers"],
            mlp_config=config["nn_config"],
            task_dict=task_dict,
            device=config["device"])

        with open(os.path.join(model_name_or_path, "best_model.pt"), 'rb') as checkpt:
            model_state, optimizer_state = torch.load(checkpt)
        model.load_state_dict(model_state)

    if training_args.is_distributed:
        model = DistributedDataParallel(model)

    # Use our modified Trainer
    tune_trainer = ArgStrTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=config["data_collator"],
        task_name=config["task_name"]
    )

    # log parameters of interest to mlflow
    mlflow.log_param("bert_model", config["bert_arch"])
    mlflow.log_param("bert_hidden_layers", config["bert_hidden_layers"])
    mlflow.log_param("nn_config", config["nn_config"])
    mlflow.log_param("dropout", config["dropout_prob"])
    mlflow.log_param("weighted_dataset_loss", config["dataset_loss_method"])
    mlflow.log_param("learning rate", config["learning_rate"])
    mlflow.log_param("weight_decay", config["weight_decay"])
    mlflow.log_param("train_batch_size", config["train_batch_size"])
    mlflow.log_param("eval_batch_size", config["eval_batch_size"])
    mlflow.log_param("max_seq_length_perc", config["max_seq_length_perc"])

    tune_trainer.train_model(optimizer_state=optimizer_state, device=config["device"])


class CustomStopper(tune.Stopper):

    def __init__(self, max_iter=150):
        self.should_stop = False
        self.max_iter = max_iter

    def __call__(self, trial_id, result):
        if not self.should_stop and result["avg_pearson"] > 0.75:
            self.should_stop = True
        return self.should_stop or result["training_iteration"] >= self.max_iter

    def stop_all(self):
        return self.should_stop


def trial_name_string(trial):
    """
    :param trial: Trial
    Returns:
        trial_name (str): String representation of Trial.
    """
    return str(trial)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    ray.init(_temp_dir="/mnt/data2/Sid/tmp")

    # define the experiment parameters
    bert_arch = "bert-base-uncased"
    task_name = "STLAS_only_gretz"
    exp_name = task_name + "_v1_bb"

    # define the path from where the task data is loaded from.
    task_data_dir = "/mnt/data2/Sid/arg_quality/pytorch/task4/data/*.csv"

    # define config
    config = {
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
        "eval_batch_size": 128,
        "train_batch_size": tune.choice([64]),
        "max_seq_length": None,
        "max_seq_length_perc": 0.95,
        "data_collator": Data_Collator,

        "dropout_prob": tune.choice([0.2]),
        "bert_hidden_layers": tune.choice([4]),
        "nn_config": tune.choice([2]),
        "dataset_loss_method": tune.choice(["weighted"]),

        "learning_rate": tune.uniform(1e-6, 7e-6),
        "weight_decay": tune.uniform(0.00, 0.35),
        "num_epochs": 20,
        "max_steps": -1,  # We use num_epochs instead.

        "mlflow": {
            "experiment_name": exp_name,
            "tracking_uri": "http://mlflow.dbs.ifi.lmu.de:5000"
        }
    }

    # define scheduler
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,
        hyperparam_mutations={
            "weight_decay": lambda: random.uniform(0.0, 0.35),
            "learning_rate": lambda: random.uniform(1e-6, 9e-6),
            "dropout_prob": [0.2],
            "dataset_loss_method": ["weighted"],
            # "bert_hidden_layers": [3, 4],
            # "nn_config": [1, 2, 3],
        })
    # define command-line reporter
    reporter = CLIReporter(
        parameter_columns={
            "bert_hidden_layers": "bert_hidden_layers",
            "nn_config": "nn_config",
            "dropout_prob": "dropout_prob",
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "train_batch_size": "train_batch_size",
            "dataset_loss_method": "dataset_loss"
        },
        metric_columns=[
            "epoch", "gretz_pearson",
            "toledo_pearson", "swanson_pearson", "ukp_pearson", "avg_pearson"
        ],
        max_report_frequency=450)

    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name)
    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    mlflow.set_experiment(experiment_name=exp_name)

    stopper = CustomStopper(max_iter=90)

    if config["is_distributed"]:
        train_bert = DistributedTrainableCreator(
            train_bert,
            num_workers=2,
            num_cpus_per_worker=8,
            num_gpus_per_worker=2,
        )
    analysis = tune.run(
        train_bert,
        resources_per_trial=config["resources_per_trial"],
        config=config,
        num_samples=2,
        scheduler=scheduler,
        resume=False,
        reuse_actors=False,
        stop=stopper,
        metric="avg_pearson",
        mode="max",
        verbose=2,
        keep_checkpoints_num=3,
        checkpoint_score_attr="avg_pearson",
        progress_reporter=reporter,
        local_dir="./ray_results/",
        trial_dirname_creator=trial_name_string,
        name=exp_name)

    best_checkpoint = recover_checkpoint(
        str(analysis.get_best_trial(metric="avg_pearson", mode="max").checkpoint.value))
    best_config = analysis.get_best_config(metric="avg_pearson", mode="max")
    best_config["best_checkpoint_path"] = best_checkpoint
    print("Best Config details:", best_config)

    best_model_path_name = "./best_model_details_" + exp_name + ".json"

    with open(best_model_path_name, 'w') as f:
        f.write(json.dumps(best_config, default=lambda x: '<not serializable>'))


# :TODO: 3. Write inference step for STLAS? ~ not needed right now
# :TODO: 5. correct truncation and distributed computing
