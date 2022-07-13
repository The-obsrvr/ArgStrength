"""
Performs inference using trained model.
"""
import os
from argparse import ArgumentParser, Namespace
import yaml
import random
from statistics import mean, stdev

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything

from regressor import ArgStrRanker
from datasets import ArgumentDataset


def load_model_from_experiment(experiment_folder: str):
    """Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    trained_model = ArgStrRanker.load_from_checkpoint(
        checkpoint_path, hparams=Namespace(**hparams)
    )
    # Make sure model is in prediction mode
    trained_model.eval()
    trained_model.freeze()
    trained_model.apply(trained_model.apply_dropout)

    return trained_model


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiments/version_20-02-2022--14-51-31/",
        help="Path to the experiment folder.",
    )
    # gpu args
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu id to use.'")
    parser.add_argument(
        "--num_of_inference_runs",
        type=int,
        default=15,
        help="Number of times the inference loop is repeated.",
    )
    parser.add_argument(
        "--source_known",
        type=bool,
        default=False,
        help="Whether source of the argument is known or not.",
    )
    parser.add_argument(
        "--predict_data_folder",
        type=str,
        default="/mnt/data2/Sid/arg_quality/pytorch/pytorch_lightning/Data/*.csv",
        help="Path to the predict data folder.",
    )
    parser.add_argument(
        "--report_metrics",
        type=bool,
        default=False,
        help=(
            "whether to report the performance metrics. Default: False"
        )
    )
    parser.add_argument(
        "--infer_results_path",
        default="/mnt/data2/Sid/arg_quality/pytorch/pytorch_lightning/infer_results.csv",
        type=str,
        help="Path to the file containing the evaluation results.",
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="mean",
        choices=["mean", "var", "wt-var", "all"],
        help=(
            "What aggregation method to use for logits aggregation. Default: mean"
        )
    )

    predict_hparams = parser.parse_args()

    print("Loading model...")
    model = load_model_from_experiment(predict_hparams.experiment)
    os.environ["CUDA_VISIBLE_DEVICES"] = predict_hparams.gpus
    device = torch.device('cuda'
                          if torch.cuda.is_available()
                          else 'cpu')
    model.to(device)
    # print(model)

    # ------------------
    # Start Prediction
    # ------------------
    # 1. Load Prediction Dataset:
    # get argument encodings for the predict datasets.

    model.hparams.max_Seq_length = None
    predict_dataset = ArgumentDataset(model.hparams,
                                      data_folder=predict_hparams.predict_data_folder,
                                      mode="predict",
                                      source_known=False,
                                      prepare_target=False,
                                      tokenizer=model.tokenizer,
                                      return_files=True
                                      )
    # get the labels and dataset information as well from the predict datasets.
    predict_dataset1 = ArgumentDataset(model.hparams,
                                       data_folder=predict_hparams.predict_data_folder,
                                       mode="predict",
                                       tokenizer=model.tokenizer,
                                       )
    test_filenames = predict_dataset.file_names
    print("Test Files: ", test_filenames)
    print("Length of test set: ", len(predict_dataset1.data))
    # print(len(predict_dataset.data))

    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        sampler=None,
        batch_size=64,
        collate_fn=model.prepare_sample,
        num_workers=model.hparams.loader_workers
    )

    # 2. Set loop for 20 seeds for different dropouts.

    logits_dict_all_seeds = []

    seed_list = random.sample(range(0, 10000), predict_hparams.num_of_inference_runs)

    for seed_value in seed_list:
        seed_everything(seed_value)
        logits_dicts_by_seed = {train_dataset: [] for train_dataset in model.task_list}

        for _, batch in enumerate(predict_dataloader):

            with torch.no_grad():
                logits_dict_by_batch = model.predict(batch, device)

                for train_dataset_name in model.task_list:
                    logits_dicts_by_seed[train_dataset_name].extend(logits_dict_by_batch[train_dataset_name].tolist())

        #  flatten the list of lists.
        for train_dataset_name in model.task_list:
            logits_dicts_by_seed[train_dataset_name] = [round(item, 4)
                                                        for sublist in logits_dicts_by_seed[train_dataset_name]
                                                        for item in sublist]
        # print(len(logits_dicts_by_seed["gretz"]))
        logits_dict_all_seeds.append(logits_dicts_by_seed)

    logits_df = pd.DataFrame(logits_dict_all_seeds)
    # print(logits_df.head())

    # 3. Aggregate the logit values over the seeds: take average of the values.
    logits_aggregated_over_seeds_mean = {}
    logits_aggregated_over_seeds_std = {}

    for trained_dataset_name in model.task_list:
        # loop over each seed entry and extract the specified dataset logit values.
        data_set_logits = [logits_df[trained_dataset_name][i] for i in range(len(logits_df))]
        #  take the mean over the seeds and round the value.
        data_set_mean_logits = list(map(mean, zip(*data_set_logits)))
        data_set_mean_logits = list(map(
            lambda x: round(x, ndigits=6), data_set_mean_logits))
        logits_aggregated_over_seeds_mean[trained_dataset_name] = data_set_mean_logits

        # for var method:
        if "var" in predict_hparams.aggregation_method:
            data_set_std_logits = list(map(stdev, zip(*data_set_logits)))
            data_set_std_logits = list(map(
                lambda x: round(x, ndigits=6), data_set_std_logits))
            logits_aggregated_over_seeds_std[trained_dataset_name] = data_set_std_logits

    assert len(logits_aggregated_over_seeds_mean[model.task_list[0]]) == \
        len(logits_aggregated_over_seeds_mean[model.task_list[1]])

    # 4. Aggregate the logits over the dataset values: based on the aggregation method.
    logits_means_df = pd.DataFrame(logits_aggregated_over_seeds_mean)
    # print(logits_means_df.head())
    logits_vars_df = pd.DataFrame(logits_aggregated_over_seeds_std) \
        if "var" in predict_hparams.aggregation_method else None

    final_logits = []
    # for mean method:
    if predict_hparams.aggregation_method == "mean":
        final_logits = logits_means_df.mean(axis=1)
        # print(final_logits)

    # for var method
    elif predict_hparams.aggregation_method == "var":
        final_logits = [logits_means_df[col_name][i] for i, col_name in
                        enumerate(logits_vars_df.idxmin(axis=1))]

    # for weighted var method
    elif predict_hparams.aggregation_method == "wt-var":
        logits_wtvars_df = pd.DataFrame()
        # generate weights from the variance values
        final_sum = []
        for i in range(len(logits_vars_df)):
            summed_row = 0
            for col in logits_vars_df:
                summed_row += 1 / logits_vars_df[col][i]
            final_sum.append(summed_row)
        for col in logits_vars_df.columns:
            logits_wtvars_df[col] = [(1 / logits_vars_df[col][i]) /
                                     final_sum[i] for i in range(len(logits_vars_df))]
        # multiply the weights with the mean values and sum
        for i in range(len(logits_wtvars_df)):
            sum_logits = 0
            for col in logits_wtvars_df.columns:
                sum_logits += logits_wtvars_df[col][i] * logits_means_df[col][i]
            final_logits.append(sum_logits)

    else:
        print("No proper aggregation method provided.")
        # for max method:
        # final_logits = logits_means_df.max(axis=1)

    # --------------------
    # 5. Calculate performance metrics.
    # --------------------
    # print(predict_dataset1.data.dataset)
    assert len(final_logits) == len(predict_dataset1.data.label)

    predictions = {"logits": final_logits,
                   "datasets": list(predict_dataset1.data.dataset)}
    targets = {"labels": list(predict_dataset1.data.label)}

    _, pearson_by_dataset, spearman_by_dataset = model.cal_performance(predictions,
                                                                       targets,
                                                                       return_by_dataset=True,
                                                                       log_metrics=False)

    # 6. save infer metrics
    results = {}
    for test_name in test_filenames:
        results[test_name + "_pearson"] = round(pearson_by_dataset[test_name].item(), 4)
        results[test_name + "_spearman"] = round(spearman_by_dataset[test_name].item(), 4)
    results["task_name"] = model.hparams.task_name
    results["sampling"] = model.hparams.sampling_strategy
    results["aggregation_method"] = predict_hparams.aggregation_method
    infer_results = pd.DataFrame([results], index=[0])
    if not os.path.exists(predict_hparams.infer_results_path):
        infer_results.to_csv(predict_hparams.infer_results_path, index=False)
    else:
        infer_results.to_csv(predict_hparams.infer_results_path, index=False, mode='a', header=False)

    # report results.
    if predict_hparams.report_metrics:
        print(results)
