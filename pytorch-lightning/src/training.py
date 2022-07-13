"""
Trains a model on a single node across N-gpus.
"""
import argparse
import os
import glob
import random
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
import pandas as pd
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
#     TuneReportCheckpointCallback

from regressor import ArgStrRanker


def main(hparams) -> None:
    """

    :param hparams:
    :return:
    """
    # :TODO: update function to accomodate ray tune.
    #  initialize model and data
    model = ArgStrRanker(hparams)

    # initialize early stopping
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=hparams.delta,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    # Tensorboard Callback
    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )
    loggers = [tb_logger]
    if hparams.use_mlflow:
        mlflow_logger = MLFlowLogger(experiment_name=hparams.mlflow_exp_name,
                                     run_name=hparams.task_name + "_" + str(hparams.run_seed),
                                     tracking_uri=hparams.mlflow_exp_uri)
        loggers.append(mlflow_logger)

    # Model Checkpoint Callback
    ckpt_path = os.path.join("experiments/", str(tb_logger.version), "checkpoints")

    # 4 Initialize checkpoint callback
    # -------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=False,
        monitor=hparams.monitor,
        mode=hparams.metric_mode,
        save_weights_only=True,
    )

    # Set gpus
    if torch.cuda.is_available() and hparams.gpus != -1:
        # use the specified gpu unit.
        gpus = [hparams.gpus]
    elif torch.cuda.is_available() and hparams.gpus == -1:
        # use all available gpu units
        gpus = -1
    else:
        # GPU unavailable, use CPU.
        gpus = None
    # ------------------------
    # Initialize trainer
    # ------------------------
    trainer = Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=6,
        fast_dev_run=False,
        gpus=gpus,
        deterministic=True,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        strategy=None
    )
    # ------------------------
    # Start training
    # ------------------------
    trainer.fit(model, model.data)

    # ------------------------
    # Start testing
    # ------------------------
    if hparams.perform_test:
        results = trainer.test(model=model,
                               datamodule=model.data)
        # print(results)
        results_dict = {"task_id": hparams.task_name + "_topic" if hparams.use_topic else hparams.task_name,
                        "sampling": hparams.sampling_strategy,
                        "checkpoint": ckpt_path,
                        "test_overall_pearson": round(results[0]["test_pearson_coeff"], 4)}

        for i, name in enumerate(list(results[0].keys())):
            if name != "test_pearson_coeff":
                results_dict[name] = round(results[0][name], 4)
        # sometimes certain columns are missing due to the nature of task and directly saving them creates
        # incorrect labelling. Solution:
        datasets = glob.glob(hparams.data_folder_path)
        for dataset_path in datasets:
            dataset_name = os.path.basename(dataset_path)[:-4]
            if not dataset_name + "_pearson" in results_dict.keys():
                results_dict[dataset_name + "_pearson"] = None
            if not dataset_name + "_spearman" in results_dict.keys():
                results_dict[dataset_name + "_spearman"] = None

        eval_results = pd.DataFrame([results_dict], index=[0])

        # store and append the run metrics to the csv. Create a new one if it doesn't exist.
        if not os.path.exists(hparams.eval_path):
            eval_results.to_csv(hparams.eval_path, index=False)
        else:
            eval_results.to_csv(hparams.eval_path, index=False, mode='a', header=False)


def tuning_pbt_exp(params):
    """

    :param params:
    :return:
    """
    config = {"dropout_prob": tune.choice([0.1, 0.2]),
              "bert_hidden_layers": tune.choice([1, 2, 3, 4]),
              "nn_config": tune.choice([1, 2, 3, 4]),
              "learning_rate": tune.uniform(1e-6, 7e-6),
              "weight_decay": tune.uniform(0.00, 0.35),
              }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,
        hyperparam_mutations={
            "weight_decay": lambda: random.uniform(0.0, 0.35),
            "learning_rate": lambda: tune.loguniform(1e-6, 1e-5),
            "dropout_prob": [0.2],
        })
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

    train_fn_with_parameters = tune.with_parameters(main,
                                                    hparams=params)
    resources_per_trial = {"cpu": 4,
                           "gpu": 1}
    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=hparams.num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name="tune_ptl_argq"
                        )
    print("Best hyperparameters found were: ", analysis.best_config)

    return analysis.best_config


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Argument Strength Transformer",
        add_help=True,
    )
    parser.add_argument("--seed_list",
                        default=[1, 500, 10000],
                        nargs='+',
                        help="Training/Inference seed(s).")
    parser.add_argument("--run_seed",
                        type=int,
                        default=None,
                        help="Current training run seed seed.")
    parser.add_argument(
        "--task_name",
        default="STLAS",
        required=True,
        type=str,
        help="Task name to use.",
    )
    parser.add_argument(
        "--use_topic",
        dest="use_topic",
        action='store_true',
        default=False,
        help="Whether to also feed topic information to the BERT model."
    )

    # gpu args
    parser.add_argument("--gpus",
                        type=int,
                        default=-1,
                        help="How many gpus. Default: '-1 (use all available)'")

    # MLFlow Arguments
    parser.add_argument(
        "--use_mlflow",
        dest="use_mlflow",
        action="store_true",
        default=True,
        help="Whether to use MLFlow logging."
    )
    parser.add_argument(
        "--mlflow_exp_name",
        default="lightning_ArgQ_Gen",
        type=str,
        help="Set the name of the MLFlow experiment"
    )
    parser.add_argument(
        "--mlflow_exp_uri",
        default="http://mlflow.dbs.ifi.lmu.de:5000",
        type=str,
        help="Set the URI of the MLFlow experiment"
    )

    # Ray tune Arguments
    parser.add_argument(
        "--use_tuning",
        default=False,
        type=bool,
        help="Whether to use Ray Tune for hyperparameter optimization."
    )
    parser.add_argument(
        "--num_samples",
        default=10,
        type=int,
        help="Number of samples to run while searching.",
    )
    parser.add_argument(
        "--config_file_path",
        default=None,
        type=str,
        help="Path to config file containing hoptimized hyperparameters"
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_loss", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--delta",
        default=0.001,
        type=float,
        help=(
            "minimum delta value for early stopping callback."
        ),
    )

    #  Testing Arguments
    parser.add_argument(
        "--perform_test",
        type=bool,
        default=True,
        help=(
            "whether to perform testing. Default: True"
        )
    )
    parser.add_argument(
        "--eval_path",
        default="evaluation_results_balanced.csv",
        type=str,
        help="Path to the file containing the evaluation results.",
    )
    parser.add_argument(
        "--report_metrics",
        type=bool,
        default=False,
        help=(
            "whether to report the metrics (aggregated over the seeds). Default: True"
        )
    )

    # Training Arguments
    parser.add_argument(
        "--perform_training",
        type=bool,
        default=True,
        help=(
            "whether to perform training. Default: True"
        )
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=50,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=2,
        type=int,
        help=(
            "Sanity check runs n validation batches before starting the training routine."
            " Set it to -1 to run all batches in all validation dataloaders.."
        ),
    )

    # Data Related Arguments
    parser.add_argument(
        "--data_folder_path",
        default="/mnt/data2/Sid/arg_quality/pytorch/pytorch_lightning/Data/*.csv",
        type=str,
        help="Path to the file containing the data.",
    )

    # Read the Arguments.
    # Each experiment defines model-specific arguments
    parser = ArgStrRanker.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # Run Optimization if required.
    # ---------------------
    if hparams.use_tuning:
        best_config = tuning_pbt_exp(params=hparams)
    # TODO    save the config data to a file.

    # ---------------------
    # RUN TRAINING: without hyperparameter optimization
    # ---------------------
    else:
        #  Check if config file path exists. If so, read it and update hparams with optimized values.
        if hparams.config_file_path is not None and os.path.exists(hparams.config_file_path):
            # :TODO: read config file and update hparams.
            # read config file
            # update hparams.
            hparams.learning_rate = 1.0

        if hparams.run_seed is not None:
            #  single seed run
            seed_everything(seed=hparams.run_seed)
            main(hparams)
        else:
            #  for multiple seed runs
            for _, seed in enumerate(hparams.seed_list):
                seed_everything(seed=seed)
                hparams.run_seed = seed
                #  Start run.
                main(hparams)

        # ---------------------
        #  Report metrics
        # ---------------------
        if hparams.report_metrics:
            random_results = pd.read_csv(hparams.eval_path)
            random_results_agg = random_results.groupby(["task_id", "sampling"], as_index=False).agg(
                {
                    'gretz_pearson': ['mean', 'std'],
                    'toledo_pearson': ['mean', 'std'],
                    'swanson_pearson': ['mean', 'std'],
                    'webis_pearson': ['mean', 'std'],
                    'ukp_pearson': ['mean', 'std'],
                    'gretz_spearman': ['mean', 'std'],
                    'toledo_spearman': ['mean', 'std'],
                    'swanson_spearman': ['mean', 'std'],
                    'webis_spearman': ['mean', 'std'],
                    'ukp_spearman': ['mean', 'std']
                })

            print(random_results_agg)

    # :TODO add in ray tune - currently doing.
