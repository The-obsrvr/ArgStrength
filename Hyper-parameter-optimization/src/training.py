from __future__ import absolute_import, division, print_function

# Local Imports
from utils import pearson_and_spearman, set_seed
from arguments import TrainingArguments

# Standard Imports
import os
import random
import ast
import json
from collections import defaultdict
from typing import Tuple, List
from statistics import mean, stdev
from scipy.stats import pearsonr, spearmanr

# Third Party Imports
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.integration.torch import distributed_checkpoint_dir
import torch
from torch import nn
from transformers import (get_linear_schedule_with_warmup, AdamW, DataCollator, Trainer)
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import mlflow

PREFIX_CHECKPOINT_DIR = "best_model"


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0, save_model: bool = True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_model (bool): whether the model should be saved or not.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_model = save_model

    def __call__(self, val_loss, model, optimizer, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model, optimizer, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(val_loss, model, optimizer, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, dir_path):
        """
        Saves model when validation loss decrease.
        :param optimizer:
        :param dir_path:
        :param val_loss:
        :param model:
        :return:
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model with best "
                  "val score...")
        self.val_loss_min = val_loss
        model_path = os.path.join(dir_path, "best_model.pt")
        torch.save((model.state_dict(), optimizer.state_dict()), model_path)


class ArgStrTrainer(Trainer):

    def __init__(self, model, args: TrainingArguments, train_dataset=None, eval_dataset=None, test_dataset=None,
                 data_collator=None, task_name: str = None):
        super().__init__(model)
        """
        Initialization of Main class containing methods for training a Multi Task Argument Strength Ranker

        :param model:
            The (pre-trained) model used for training.
        :param args (TrainingArguments):
            The Training Arguments
        :param train_dataset:
            The dataset for training
        :param eval_dataset:
            The dataset for validation
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.model = model
        self.global_step = 0
        self.dataset_weights = None
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DataCollator()
        self.task_name = task_name
        # for inference
        self.infer_df = pd.DataFrame()
        self.final_logits = []

    def get_dataloader(self, data_set, is_train: bool = False, set_batch_size: int = None) -> DataLoader:
        """
        Creates a Data loader containing the validation dataset using a Weighted Data Sampler.
        :param set_batch_size:
        :param data_set:
        :param is_train:
        :returns:
            Data loader containing the validation dataset.
        """
        if is_train and self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_sampler = None
        if is_train:
            data_sampler = RandomSampler(data_set)
        if set_batch_size is None:
            batch_size = self.args.train_batch_size if is_train else self.args.eval_batch_size
        else:
            batch_size = set_batch_size
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=data_sampler,
            drop_last=True,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def set_weights_to_dataset(self, complete_data_set):
        """
        set weights to the datasets based on the method called (equal/weighted/no weights)
        :return: weights for each dataset
        """
        # define the weights for each dataset loss (Set for training)
        split_features_dict = defaultdict(list)
        for feature in complete_data_set:
            split_features_dict[feature.data_set].append(feature)
        dataset_weights = {key: 0 for key in split_features_dict.keys()}

        for dataset, features_list in split_features_dict.items():
            if self.args.weighted_dataset_loss == "weighted":
                dataset_weights[dataset] = round(
                    len(complete_data_set) / (len(split_features_dict.keys()) * len(features_list)), 3)
            elif self.args.weighted_dataset_loss == "unweighted":
                dataset_weights[dataset] = 1
            else:  # default: args.dataset_loss_method == "equal"
                dataset_weights[dataset] = (1 / len(split_features_dict.keys()))

        self.dataset_weights = dataset_weights

    def _evaluate_model(self, device: torch.device,
                        batch_dataloader: DataLoader,
                        mode: str,
                        epoch: int,
                        mlflow_logging: bool = True) -> Tuple:
        """

        :param device:
        :param batch_dataloader:
        :param mode:
        :return:
        """

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()
        # Tracking variables
        total_eval_loss = 0
        gretz_eval_batch_losses = []
        toledo_eval_batch_losses = []
        swanson_eval_batch_losses = []
        ukprank_eval_batch_losses = []
        gretz_logits = []
        gretz_labels = []
        swanson_logits = []
        swanson_labels = []
        toledo_logits = []
        toledo_labels = []
        ukp_logits = []
        ukp_labels = []
        # Evaluate data for one epoch
        for batch in batch_dataloader:
            batch.move_tensors_to_device(device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                bert_batch_output = self.model(batch)

            weighted_loss = bert_batch_output.calculate_weighted_loss(self.dataset_weights)

            # Accumulate the validation loss.
            total_eval_loss += weighted_loss.item()

            for single_ds_output in bert_batch_output.bert_single_dataset_outputs:
                # Move logits and labels to CPU
                cpu_logits_list = single_ds_output.logits.detach().cpu().numpy()
                cpu_labels_list = single_ds_output.labels.to('cpu').numpy()

                cpu_logits_list_transposed = np.ndarray.transpose(cpu_logits_list)
                cpu_logits_list_transposed = cpu_logits_list_transposed[0]

                # store the logits, labels and the validation loss for each respective data set.
                if single_ds_output.data_set == "gretz":
                    gretz_eval_batch_losses.append(single_ds_output.loss)
                    gretz_logits.extend(cpu_logits_list_transposed)
                    gretz_labels.extend(cpu_labels_list)
                elif single_ds_output.data_set == "swanson":
                    swanson_eval_batch_losses.append(single_ds_output.loss)
                    swanson_logits.extend(cpu_logits_list_transposed)
                    swanson_labels.extend(cpu_labels_list)
                elif single_ds_output.data_set == "toledo":
                    toledo_eval_batch_losses.append(single_ds_output.loss)
                    toledo_logits.extend(cpu_logits_list_transposed)
                    toledo_labels.extend(cpu_labels_list)
                else:
                    ukprank_eval_batch_losses.append(single_ds_output.loss)
                    ukp_logits.extend(cpu_logits_list_transposed)
                    ukp_labels.extend(cpu_labels_list)

        # Tracking metrics for each dataset that we use for logging to MlFlow.
        # We check to see which dataset exists in the current setup and accordingly calculate the
        # corresponding evaluation metrics or just set them to zero.
        if len(gretz_logits) != 0:
            gretz_pearson, gretz_spearman, _ = pearson_and_spearman(gretz_logits, gretz_labels)
            avg_gretz_eval_loss = sum(gretz_eval_batch_losses) / len(gretz_eval_batch_losses)
        else:
            gretz_pearson, gretz_spearman, avg_gretz_eval_loss = 0, 0, 0
        if len(swanson_logits) != 0:
            swanson_pearson, swanson_spearman, _ = pearson_and_spearman(swanson_logits, swanson_labels)
            avg_swanson_eval_loss = sum(swanson_eval_batch_losses) / len(swanson_eval_batch_losses)
        else:
            swanson_pearson, swanson_spearman, avg_swanson_eval_loss = 0, 0, 0
        if len(toledo_logits) != 0:
            toledo_pearson, toledo_spearman, _ = pearson_and_spearman(toledo_logits, toledo_labels)
            avg_toledo_eval_loss = sum(toledo_eval_batch_losses) / len(toledo_eval_batch_losses)
        else:
            toledo_pearson, toledo_spearman, avg_toledo_eval_loss = 0, 0, 0
        if len(ukp_logits) != 0:
            ukp_pearson, ukp_spearman, _ = pearson_and_spearman(ukp_logits, ukp_labels)
            avg_ukprank_eval_loss = sum(ukprank_eval_batch_losses) / len(ukprank_eval_batch_losses)
        else:
            ukp_pearson, ukp_spearman, avg_ukprank_eval_loss = 0, 0, 0

        # Calculate the average loss over all of the batches.
        avg_eval_loss = total_eval_loss / len(batch_dataloader)

        if mode == "validation":
            # Check to see which datasets have been used and log metrics accordingly.
            if mlflow_logging:
                mlflow.log_metric("avg_val_loss", avg_eval_loss, epoch)
                if "_LOO_toledo" in self.task_name:
                    mlflow.log_metric("avg_gretz_val_loss", avg_gretz_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_swanson_val_loss", avg_swanson_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_ukprank_val_loss", avg_ukprank_eval_loss.item(), epoch)
                elif "_LOO_gretz" in self.task_name:
                    mlflow.log_metric("avg_toledo_val_loss", avg_toledo_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_swanson_val_loss", avg_swanson_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_ukprank_val_loss", avg_ukprank_eval_loss.item(), epoch)
                elif "_LOO_swanson" in self.task_name:
                    mlflow.log_metric("avg_gretz_val_loss", avg_gretz_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_toledo_val_loss", avg_toledo_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_ukprank_val_loss", avg_ukprank_eval_loss.item(), epoch)
                elif "_LOO_ukp" in self.task_name:
                    mlflow.log_metric("avg_gretz_val_loss", avg_gretz_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_toledo_val_loss", avg_toledo_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_swanson_val_loss", avg_swanson_eval_loss.item(), epoch)
                elif "only_gretz" in self.task_name:
                    mlflow.log_metric("avg_gretz_val_loss", avg_gretz_eval_loss.item(), epoch)
                elif "only_toledo" in self.task_name:
                    mlflow.log_metric("avg_toledo_val_loss", avg_toledo_eval_loss.item(), epoch)
                elif "only_swanson" in self.task_name:
                    mlflow.log_metric("avg_swanson_val_loss", avg_swanson_eval_loss.item(), epoch)
                elif "only_ukp" in self.task_name:
                    mlflow.log_metric("avg_ukprank_val_loss", avg_ukprank_eval_loss.item(), epoch)
                else:
                    mlflow.log_metric("avg_gretz_val_loss", avg_gretz_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_toledo_val_loss", avg_toledo_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_swanson_val_loss", avg_swanson_eval_loss.item(), epoch)
                    mlflow.log_metric("avg_ukprank_val_loss", avg_ukprank_eval_loss.item(), epoch)

            return avg_eval_loss, avg_gretz_eval_loss, avg_toledo_eval_loss, avg_swanson_eval_loss, \
                   avg_ukprank_eval_loss
        else:
            if mlflow_logging:
                mlflow.log_metric("avg_test_loss", avg_eval_loss, epoch)
                if "MTLAS_LOO_gretz" not in self.task_name:
                    mlflow.log_metric("avg_gretz_test_loss", avg_gretz_eval_loss.item(), epoch)
                    mlflow.log_metric("gretz pearson", gretz_pearson.item(), epoch)
                    mlflow.log_metric("gretz spearman", gretz_spearman.item(), epoch)
                if "MTLAS_LOO_toledo" not in self.task_name:
                    mlflow.log_metric("avg_toledo_test_loss", avg_toledo_eval_loss.item(), epoch)
                    mlflow.log_metric("toledo pearson", toledo_pearson.item(), epoch)
                    mlflow.log_metric("toledo spearman", toledo_spearman.item(), epoch)
                if "MTLAS_LOO_swanson" not in self.task_name:
                    mlflow.log_metric("avg_swanson_test_loss", avg_swanson_eval_loss.item(), epoch)
                    mlflow.log_metric("swanson pearson", swanson_pearson.item(), epoch)
                    mlflow.log_metric("swanson spearman", swanson_spearman.item(), epoch)
                if "MTLAS_LOO_ukp" not in self.task_name:
                    mlflow.log_metric("avg_ukprank_test_loss", avg_ukprank_eval_loss.item(), epoch)
                    mlflow.log_metric("ukp pearson", ukp_pearson.item(), epoch)
                    mlflow.log_metric("ukp spearman", ukp_spearman.item(), epoch)

            return avg_eval_loss, gretz_pearson, toledo_pearson, swanson_pearson, ukp_pearson

    def train_model(self, device: torch.device,
                    optimizer_state=None,
                    mlflow_logging: bool = True,
                    retraining: bool = False,
                    seed_value: int = 3):
        """
        Main data training method. We use a pre-trained BERT model with a single linear classification layer on top.
        Uses early stopping and returns the best model trained. (Maximum number of epochs prevents infinite runtime).

        :param seed_value:
        :param retraining:
        :param mlflow_logging:
        :param optimizer_state:
            The saved optimizer state.
        :param device:
            The device to train on.
        :returns:
            The best model trained after stopping early.
        """
        save_model = True
        tune_report = True
        eval_path = None
        if retraining:
            save_model = False
            tune_report = False
            eval_path = "./randomized_results/randomized_results2.csv"

        # Get the data loaders
        train_dataloader = self.get_dataloader(self.train_dataset, is_train=True)
        validation_dataloader = self.get_dataloader(self.eval_dataset)
        test_dataloader = self.get_dataloader(self.test_dataset)

        # Move model to the training device
        self.model.to(device)

        # Initialize optimizer and load the previous optimizer.
        optimizer = AdamW(self.model.parameters(),
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon
                          )
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        # Total number of training steps
        total_steps = len(train_dataloader) * self.args.max_num_train_epochs

        # Learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        # total_t0 = time.time()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, save_model=save_model)
        early_stop = False

        avg_train_loss, avg_val_loss, avg_test_loss, gretz_pearson, toledo_pearson, \
        ukp_pearson, swanson_pearson, avg_test_pearson = 0, 0, 0, 0, 0, 0, 0, 0

        # We loop epochs until a max, however early stopping should end training much sooner!
        for epoch in tqdm(range(self.args.max_num_train_epochs)):

            # do not start a new epoch if the early stop criterion is met
            if early_stop:
                break

            # Reset the total loss for this epoch.
            total_train_loss = 0
            gretz_train_batch_losses = []
            toledo_train_batch_losses = []
            swanson_train_batch_losses = []
            ukprank_train_batch_losses = []

            # Put the model into training mode.
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader, start=1):

                # Move batch to training device
                batch.move_tensors_to_device(device)

                # Clear any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass.
                bert_batch_output = self.model(batch)

                # Calculate the weighted loss based on the result
                weighted_loss = bert_batch_output.calculate_weighted_loss(self.dataset_weights)
                # Perform a backward pass to calculate the gradients.
                weighted_loss.backward()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end.
                total_train_loss += weighted_loss.item()

                # Accumulate training loss for each dataset.
                for bert_single_dataset_output in bert_batch_output.bert_single_dataset_outputs:
                    if bert_single_dataset_output.data_set == "gretz":
                        gretz_train_batch_losses.append(bert_single_dataset_output.loss)
                    elif bert_single_dataset_output.data_set == "swanson":
                        swanson_train_batch_losses.append(bert_single_dataset_output.loss)
                    elif bert_single_dataset_output.data_set == "toledo":
                        toledo_train_batch_losses.append(bert_single_dataset_output.loss)
                    else:
                        ukprank_train_batch_losses.append(bert_single_dataset_output.loss)

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                self.global_step = step

                # Validation after every epoch
                if step == len(train_dataloader) and self.args.evaluate_during_training:

                    # Calculate the average validation metrics
                    avg_val_loss, avg_gretz_val_loss, avg_toledo_val_loss, avg_swanson_val_loss, avg_ukprank_val_loss = \
                        self._evaluate_model(device, validation_dataloader, mode="validation",
                                             epoch=epoch, mlflow_logging=mlflow_logging)

                    # Calculate the average test metrics
                    avg_test_loss, gretz_pearson, toledo_pearson, swanson_pearson, ukp_pearson = \
                        self._evaluate_model(device, test_dataloader, mode="test",
                                             epoch=epoch, mlflow_logging=mlflow_logging)

                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if tune_report and save_model:
                        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
                            # This is the directory name that Huggingface requires.
                            output_dir = os.path.join(
                                checkpoint_dir,
                                "best_model-{}".format(self.global_step))
                            os.makedirs(output_dir, exist_ok=True)
                    else:
                        if save_model:
                            output_dir = os.path.join(str(self.task_name), "best_model-{}".format(self.global_step))
                            os.makedirs(output_dir, exist_ok=True)
                        else:
                            output_dir = None
                    early_stopping(avg_val_loss, self.model, optimizer, output_dir)

                    if early_stopping.early_stop:
                        # Break out of the current epoch
                        early_stop = True
                        break  # not needed since we always through all batches and dont allow break in the middle.

            if mlflow_logging:

                # log training losses for each set in the current training run.
                if "_LOO_gretz" in self.task_name:
                    avg_toledo_train_loss = sum(toledo_train_batch_losses) / len(toledo_train_batch_losses)
                    mlflow.log_metric("avg_toledo_train_loss", avg_toledo_train_loss.item(), epoch)
                    avg_swanson_train_loss = sum(swanson_train_batch_losses) / len(swanson_train_batch_losses)
                    mlflow.log_metric("avg_swanson_train_loss", avg_swanson_train_loss.item(), epoch)
                    avg_ukprank_train_loss = sum(ukprank_train_batch_losses) / len(ukprank_train_batch_losses)
                    mlflow.log_metric("avg_ukprank_train_loss", avg_ukprank_train_loss.item(), epoch)
                elif "_LOO_toledo" in self.task_name:
                    avg_gretz_train_loss = sum(gretz_train_batch_losses) / len(gretz_train_batch_losses)
                    mlflow.log_metric("avg_gretz_train_loss", avg_gretz_train_loss.item(), epoch)
                    avg_swanson_train_loss = sum(swanson_train_batch_losses) / len(swanson_train_batch_losses)
                    mlflow.log_metric("avg_swanson_train_loss", avg_swanson_train_loss.item(), epoch)
                    avg_ukprank_train_loss = sum(ukprank_train_batch_losses) / len(ukprank_train_batch_losses)
                    mlflow.log_metric("avg_ukprank_train_loss", avg_ukprank_train_loss.item(), epoch)
                elif "_LOO_swanson" in self.task_name:
                    avg_gretz_train_loss = sum(gretz_train_batch_losses) / len(gretz_train_batch_losses)
                    mlflow.log_metric("avg_gretz_train_loss", avg_gretz_train_loss.item(), epoch)
                    avg_toledo_train_loss = sum(toledo_train_batch_losses) / len(toledo_train_batch_losses)
                    mlflow.log_metric("avg_toledo_train_loss", avg_toledo_train_loss.item(), epoch)
                    avg_ukprank_train_loss = sum(ukprank_train_batch_losses) / len(ukprank_train_batch_losses)
                    mlflow.log_metric("avg_ukprank_train_loss", avg_ukprank_train_loss.item(), epoch)
                elif "_LOO_ukp" in self.task_name:
                    avg_gretz_train_loss = sum(gretz_train_batch_losses) / len(gretz_train_batch_losses)
                    mlflow.log_metric("avg_gretz_train_loss", avg_gretz_train_loss.item(), epoch)
                    avg_toledo_train_loss = sum(toledo_train_batch_losses) / len(toledo_train_batch_losses)
                    mlflow.log_metric("avg_toledo_train_loss", avg_toledo_train_loss.item(), epoch)
                    avg_swanson_train_loss = sum(swanson_train_batch_losses) / len(swanson_train_batch_losses)
                    mlflow.log_metric("avg_swanson_train_loss", avg_swanson_train_loss.item(), epoch)
                elif "only_gretz" in self.task_name:
                    avg_gretz_train_loss = sum(gretz_train_batch_losses) / len(gretz_train_batch_losses)
                    mlflow.log_metric("avg_gretz_train_loss", avg_gretz_train_loss.item(), epoch)
                elif "only_toledo" in self.task_name:
                    avg_toledo_train_loss = sum(toledo_train_batch_losses) / len(toledo_train_batch_losses)
                    mlflow.log_metric("avg_toledo_train_loss", avg_toledo_train_loss.item(), epoch)
                elif "only_swanson" in self.task_name:
                    avg_swanson_train_loss = sum(swanson_train_batch_losses) / len(swanson_train_batch_losses)
                    mlflow.log_metric("avg_swanson_train_loss", avg_swanson_train_loss.item(), epoch)
                elif "only_ukp" in self.task_name:
                    avg_ukprank_train_loss = sum(ukprank_train_batch_losses) / len(ukprank_train_batch_losses)
                    mlflow.log_metric("avg_ukprank_train_loss", avg_ukprank_train_loss.item(), epoch)
                else:
                    avg_gretz_train_loss = sum(gretz_train_batch_losses) / len(gretz_train_batch_losses)
                    mlflow.log_metric("avg_gretz_train_loss", avg_gretz_train_loss.item(), epoch)
                    avg_toledo_train_loss = sum(toledo_train_batch_losses) / len(toledo_train_batch_losses)
                    mlflow.log_metric("avg_toledo_train_loss", avg_toledo_train_loss.item(), epoch)
                    avg_swanson_train_loss = sum(swanson_train_batch_losses) / len(swanson_train_batch_losses)
                    mlflow.log_metric("avg_swanson_train_loss", avg_swanson_train_loss.item(), epoch)
                    avg_ukprank_train_loss = sum(ukprank_train_batch_losses) / len(ukprank_train_batch_losses)
                    mlflow.log_metric("avg_ukprank_train_loss", avg_ukprank_train_loss.item(), epoch)

                # log average training loss in the current training run.
                avg_train_loss = total_train_loss / len(train_dataloader)
                mlflow.log_metric("avg_train_loss", avg_train_loss, epoch)

                pearson_array = np.array([gretz_pearson, toledo_pearson, swanson_pearson, ukp_pearson])
                avg_test_pearson = pearson_array[pearson_array != 0].mean()
                mlflow.log_metric("avg_pearson", avg_test_pearson, epoch)

                # report the metrics back to tune.
                if tune_report:
                    report_metrics = {"epoch": epoch,
                                      "avg_train_loss": avg_train_loss,
                                      "avg_val_loss": avg_val_loss,
                                      "avg_test_loss": avg_test_loss,
                                      "gretz_pearson": gretz_pearson,
                                      "toledo_pearson": toledo_pearson,
                                      "ukp_pearson": ukp_pearson,
                                      "swanson_pearson": swanson_pearson,
                                      "avg_pearson": avg_test_pearson}
                    tune.report(**report_metrics)

        if retraining:
            # save metrics from final epoch to a csv file
            eval_results = pd.DataFrame({"model_id": self.task_name,
                                         "seed": seed_value,
                                         "avg_train_loss": avg_train_loss,
                                         "avg_val_loss": avg_val_loss,
                                         "avg_test_loss": avg_test_loss,
                                         "gretz_pearson": gretz_pearson,
                                         "toledo_pearson": toledo_pearson,
                                         "ukp_pearson": ukp_pearson,
                                         "swanson_pearson": swanson_pearson,
                                         "avg_pearson": avg_test_pearson}, index=[0])
            if not os.path.exists(eval_path):
                eval_results.to_csv(eval_path, index=False)
            else:
                eval_results.to_csv(eval_path, index=False, mode='a', header=False)

    @staticmethod
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    def _aggregate_inference_logits(self, agg_method: str = "mean"):
        """

        :param agg_method: aggregation method: mean, max, var and wt-var
        :return:
        """
        self.final_logits = []
        logits_mean_set = []
        logits_var_set = []
        # loop over each dataset column
        for col_name in self.infer_df.columns[2:]:
            data_set_logits = []
            # loop over each seed entry
            for i in range(len(self.infer_df)):
                # convert the string column to a list
                dataset_logits = ast.literal_eval(self.infer_df[col_name][i])
                data_set_logits.append(dataset_logits)

            data_set_mean_logits = list(map(mean, zip(*data_set_logits)))
            data_set_mean_logits = list(map(
                lambda x: round(x, ndigits=6), data_set_mean_logits))
            logits_mean_set.append(data_set_mean_logits)

            # for var method:
            if "var" in agg_method:
                data_set_var_logits = list(map(stdev, zip(*data_set_logits)))
                data_set_var_logits = list(map(
                    lambda x: round(x, ndigits=6), data_set_var_logits))
                logits_var_set.append(data_set_var_logits)

        logits_means_df = pd.DataFrame(np.array(logits_mean_set).T, columns=self.infer_df.columns[2:])
        logits_vars_df = pd.DataFrame(np.array(logits_var_set).T, columns=self.infer_df.columns[2:])\
            if "var" in agg_method else pd.DataFrame()

        # for mean method:
        if agg_method == "mean":
            self.final_logits = logits_means_df.mean(axis=1)

        # for var method
        elif agg_method == "var":
            self.final_logits = [logits_means_df[col_name][i] for i, col_name in
                                 enumerate(logits_vars_df.idxmin(axis=1))]

        # for weighted var method
        elif agg_method == "wt-var":
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
                self.final_logits.append(sum_logits)

        # for max method:
        else:
            self.final_logits = logits_means_df.max(axis=1)

    def _cal_eval_metrics(self, agg_method: str = "mean"):
        """
        :param agg_method: aggregation method: mean, max, var and wt-var
        :return: inference_metrics dict
        """

        datasets = ast.literal_eval(self.infer_df["orig_ds"][0])
        labels = ast.literal_eval(self.infer_df["labels"][0])
        labels = list(map(lambda x: round(x, ndigits=6), labels))

        assert len(datasets) == len(labels)
        assert len(self.final_logits) == len(labels)

        inference_metrics = {}

        unlabeled_pearson = round(pearsonr(self.final_logits, labels)[0], 4)
        unlabeled_spearman = round(spearmanr(labels, self.final_logits)[0], 4)
        key_pearson = agg_method + "_unknown_pearson"
        key_spearman = agg_method + "_unknown_spearman"
        inference_metrics[key_pearson] = unlabeled_pearson
        inference_metrics[key_spearman] = unlabeled_spearman

        # group values by the datasets to help calculate the pearson values by dataset.
        data2 = pd.DataFrame(np.column_stack([datasets, labels, self.final_logits]),
                             columns=["datasets", "labels", "logits"])
        grouped_dataset_df = data2.groupby(['datasets'])

        # loop over each dataset
        for data_set in list(grouped_dataset_df.groups.keys()):
            data_gp = grouped_dataset_df.get_group(data_set)
            logits = [float(i) for i in data_gp["logits"]]
            labels = [float(i) for i in data_gp["labels"]]
            pearson_value = round(pearsonr(logits, labels)[0], ndigits=4)
            spearman_value = round(spearmanr(logits, labels)[0], ndigits=4)
            key_pearson = agg_method + "_" + data_set + "_pearson"
            key_spearman = agg_method + "_" + data_set + "_spearman"
            inference_metrics[key_pearson] = pearson_value
            inference_metrics[key_spearman] = spearman_value

        return inference_metrics

    def infer_model(self, infer_dataset,
                    device: torch.device,
                    task_dict: List = None,
                    exp_name: str = "abc",
                    exp_seed: int = 0,
                    save_logits: bool = True,
                    ):
        """

        :param save_logits:
        :param exp_seed:
        :param task_dict:
        :param exp_name:
        :param infer_dataset:
        :param device: The device to infer on.
        :returns:
            Argument strength values for the given data
        """

        if "mean" in self.task_name:
            aggregation_method = "mean"
        elif "var" in self.task_name:
            aggregation_method = "var"
        elif "wt-var" in self.task_name:
            aggregation_method = "wt-var"
        else:
            aggregation_method = "all"

        # get the dataloader
        infer_dataloader = self.get_dataloader(infer_dataset)

        # move the model to the inference device
        self.model.to(device)
        # put the model in evaluation mode during inference
        self.model.eval()
        self.model.apply(self.apply_dropout)

        # track logits values for each seed
        complete_logits_list = []

        print("Starting Inference")

        for i in range(10):
            # reset inference seed values for the dropout.
            set_seed(random.randint(0, 10000))

            # Tracking variables
            gretz_logits = []
            swanson_logits = []
            toledo_logits = []
            ukp_logits = []

            logits_dict = {"gretz": gretz_logits, "swanson": swanson_logits, "toledo": toledo_logits, "ukp": ukp_logits}
            labels_list = []
            dataset_list = []
            # Draw inference for each batch
            for batch in infer_dataloader:
                batch.move_tensors_to_device(device)

                with torch.no_grad():
                    # Forward pass, calculate the logits per head
                    bert_batch_output = self.model.forward_for_inference(batch)

                # each batch has some number of datasets. First loop is to parse through each individual dataset
                for ds_id, dataset in enumerate(bert_batch_output):

                    for idx in range(len(dataset[1][0])):
                        dataset_list.append(dataset[2])
                        labels_list.append(dataset[0][idx].to('cpu').numpy().astype(np.float32))
                        for j, reg_head in enumerate(task_dict):
                            logits_dict[reg_head].append(dataset[1][j][idx].astype(np.float32))

            labels_list = [x.tolist() for x in labels_list]
            if len(logits_dict["gretz"]) > 0:
                gretz_logits = [x.tolist() for sublist in logits_dict["gretz"] for x in sublist]
            else:
                gretz_logits = [None for sublist in logits_dict["ukp"] for _ in sublist]
            if len(logits_dict["toledo"]) > 0:
                toledo_logits = [x.tolist() for sublist in logits_dict["toledo"] for x in sublist]
            else:
                toledo_logits = [None for sublist in logits_dict["ukp"] for _ in sublist]
            if len(logits_dict["swanson"]) > 0:
                swanson_logits = [x.tolist() for sublist in logits_dict["swanson"] for x in sublist]
            else:
                swanson_logits = [None for sublist in logits_dict["ukp"] for _ in sublist]
            if len(logits_dict["ukp"]) > 0:
                ukp_logits = [x.tolist() for sublist in logits_dict["ukp"] for x in sublist]
            else:
                ukp_logits = [None for sublist in logits_dict["gretz"] for _ in sublist]

            complete_logits_list.append([dataset_list, labels_list, gretz_logits,
                                         toledo_logits, swanson_logits, ukp_logits])

        self.infer_df = pd.DataFrame(complete_logits_list,
                                     columns=["orig_ds", "labels", "gretz_logits", "toledo_logits",
                                              "swanson_logits", "ukp_logits"])

        if save_logits:
            path_name = "./infer_logits/infer_logits_" + exp_name + "_" + str(exp_seed) + ".csv"
            self.infer_df.to_csv(path_name, index=False)

        print("Calculating inference metrics")
        if save_logits:
            self.infer_df = pd.read_csv(path_name)

        if aggregation_method == "all":
            self._aggregate_inference_logits(agg_method="mean")
            inference_metrics = self._cal_eval_metrics(agg_method="mean")
            self._aggregate_inference_logits(agg_method="wt-var")
            inference_metrics.update(self._cal_eval_metrics(agg_method="wt-var"))
            self._aggregate_inference_logits(agg_method="var")
            inference_metrics.update(self._cal_eval_metrics(agg_method="var"))

        else:
            self._aggregate_inference_logits(agg_method=aggregation_method)
            inference_metrics = self._cal_eval_metrics(agg_method=aggregation_method)

        # save metrics as txt
        file_name = "./inference_results/infer_metrics_" + self.task_name + "_" + str(exp_seed) + ".json"
        with open(file_name, "w") as f:
            json.dump(inference_metrics, f, indent=4)
