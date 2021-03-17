from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Local Imports
from processors import StrengthProcessor, convert_features_to_dataset
from modeling import BertForSequenceClassification
from utils import Data_Collator, load_data_files, TestResult, RegressionResult

# Standard Imports
import numpy as np
import pandas as pd
import random
import argparse
from collections import defaultdict
from typing import Tuple, Any

# Third Party Imports
import torch
from transformers import (get_linear_schedule_with_warmup, BertTokenizer, AdamW, DataCollator)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        :param val_loss:
        :param model:
        :return:
        """
        # if self.verbose:
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def set_seed(seed=3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class Trainer:

    def __init__(self, model, args, train_dataset, eval_dataset, dataset_weights,
                 save_or_load_model_path, data_collator=None):
        """
        Initialization of Main class containing methods for training.

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
        self.args = args
        self.model = model
        self.save_or_load_model_path = save_or_load_model_path
        self.dataset_weights = dataset_weights
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DataCollator()


    def get_train_dataloader(self, batch_size: int) -> DataLoader:
        """
        Creates a Data loader containing the train dataset using a Weighted Data Sampler.
        :param batch_size:
            The size of each batch.
        :returns:
            Data loader containing the train dataset.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def get_eval_dataloader(self, batch_size) -> DataLoader:
        """
        Creates a Data loader containing the validation dataset using a Weighted Data Sampler.
        :param batch_size:
            The size of each batch.
        :returns:
            Data loader containing the validation dataset.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: validation requires a eval_dataset.")

        data_loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def _validate_data(self, device: torch.device, validation_dataloader: DataLoader) -> float:
        """
        Validation step of training. Validates on all training datasets using the validation data only.
        :param device:
            the device to validate on
        :param validation_dataloader:
            the validation data loader
        :returns:
            the average validation loss
        """
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()
        # Tracking variables
        total_eval_pearson = []
        total_eval_loss = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            batch.move_tensors_to_device(device)
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                bert_batch_output = self.model(batch)
            weighted_loss = bert_batch_output.calculate_weighted_loss(self.dataset_weights)
            # Accumulate the validation loss.
            total_eval_loss += weighted_loss.item()
            # Tracking metrics for each dataset that we use for logging to MlFlow
            pearsons = []
            for single_ds_output in bert_batch_output.bert_single_dataset_outputs:
                # Move logits and labels to CPU
                cpu_logits_list = single_ds_output.logits.detach().cpu().numpy()
                cpu_labels_list = single_ds_output.labels.to('cpu').numpy()
                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.

                # Only calculate pearson coefficient if we have more than 1 regression prediction.
                if len(cpu_logits_list) > 1:
                    cpu_logits_list_transposed = np.ndarray.transpose(cpu_logits_list)
                    cpu_logits_list_transposed = cpu_logits_list_transposed[0]
                    pearson, _, _ = pearson_and_spearman(cpu_logits_list_transposed, cpu_labels_list)
                    pearsons.append(pearson)

            if len(pearsons) > 0:
                weighted_pearson = sum(pearsons) / len(pearsons)
                total_eval_pearson.append(weighted_pearson)

        if len(total_eval_pearson) > 0:
            avg_val_pearson = sum(total_eval_pearson) / len(total_eval_pearson)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        return avg_val_loss

    def train_data(self, device: torch.device, batch_size: int):
        """
        Main data training method. We use a pre-trained BERT model with a single linear classification layer on top.
        Uses early stopping and returns the best model trained. (Maximum number of epochs prevents infinite runtime).

        :param device:
            The device to train on.
        :param batch_size:
            The batch size for training

        :returns:
            The best model trained after stopping early.
        """

        # Get the data loaders
        train_dataloader = self.get_train_dataloader(batch_size)
        validation_dataloader = self.get_eval_dataloader(batch_size)

        # Move model to the training device
        self.model.to(device)

        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(),
                          lr=self.args.learning_rate,  # args.learning_rate - default is 5e-5
                          eps=self.args.adam_epsilon  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps
        total_steps = len(train_dataloader) * self.args.max_num_train_epochs

        # Learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        # total_t0 = time.time()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stop = False

        # We loop epochs until a max, however early stopping should end training much sooner!
        for _ in tqdm(range(self.args.max_num_train_epochs)):

            # do not start a new epoch if the early stop criterion is met
            if early_stop:
                break
            #
            # logger.info('======== Epoch {:} ========'.format(epoch_i + 1, self.args.max_num_train_epochs))
            # logger.info('Starting Training... Total number of batches per epoch: {}'.format(len(train_dataloader)))
            #
            # t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode.
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader, start=1):

                # Move batch to training device
                batch.move_tensors_to_device(device)

                # Clear any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                bert_batch_output = self.model(batch)
                # Calculate the weighted loss based on the result
                weighted_loss = bert_batch_output.calculate_weighted_loss(self.dataset_weights)
                # logger.debug("    Calculated weighted loss for batch: {}".format(weighted_loss))

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end.
                total_train_loss += weighted_loss.item()

                # Perform a backward pass to calculate the gradients.
                weighted_loss.backward()

                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                # Validation after every epoch
                if step == len(train_dataloader):

                    # Calculate the average validation loss
                    avg_val_loss = self._validate_data(device, validation_dataloader)

                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(avg_val_loss, self.model)

                    if early_stopping.early_stop:
                        # logger.info("Early stopping\n")
                        # Break out of the current epoch
                        early_stop = True
                        break

        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load('checkpoint' + '.pt'))

        if self.args.save_model:
            self.model.save_pretrained(self.save_or_load_model_path)
            print("saving model")

        # logger.info("Training complete!")
        # logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


# Calculate Pearson, Spearman and Corr for Regression
def pearson_and_spearman(preds, labels) -> Tuple[float, Any, Any]:
    """Calculates the pearson and spearman correlation.

    :param preds:
    :param labels:
    :return: pearson correlation, spearman correlationn and (pearson_corr + spearman_corr) / 2
    **See Also
        stats.py
    """
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return pearson_corr, spearman_corr, (pearson_corr + spearman_corr) / 2,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_folder', default="data", type=str)
    parser.add_argument('--pretrained_name_or_path', default="bert-base-uncased", type=str)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--save_or_load_model_path', default="/mnt/data2/Sid/arg_quality/pytorch/task4/models", type=str)
    parser.add_argument('--do_eval', default=True, type=bool)
    parser.add_argument('--save_results_path', default="/mnt/data2/Sid/arg_quality/pytorch/task4", type=str)
    parser.add_argument('--do_train', default=True, type=bool)
    parser.add_argument('--dataset_loss_method', default="equal", type=str)
    parser.add_argument('--seed_list', nargs="*", type=int, default=[42])
    parser.add_argument('--max_seq_length', default=85, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max_num_train_epochs', default=25, type=int)
    parser.add_argument('--patience', default=5, type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print("Using Device:", device)
    print("GPUs:", n_gpu)

    # set up evaluation directory and path
    args.save_results_path = os.path.join(args.save_results_path, "eval_results.csv")

    for seed in args.seed_list:

        set_seed(seed)

        # set up model directory to save or load from
        save_or_load_model_path = os.path.join(args.save_or_load_model_path, "model_" +
                                                    str(args.dataset_loss_method) + str(seed))
        if not os.path.exists(save_or_load_model_path):
            os.makedirs(save_or_load_model_path)

        # initialize model
        model = BertForSequenceClassification.from_pretrained(args.pretrained_name_or_path, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_name_or_path, do_lower_case=False)

        # load the data as DataFile
        data_files = load_data_files(args.datasets_folder)
        # initialize the Processor
        processor = StrengthProcessor(data_files)

        # start training
        if args.do_train:

            # prepare the train and validation datasets
            train_dataset = convert_features_to_dataset(processor, tokenizer, args.max_seq_length, "train")
            validation_dataset = convert_features_to_dataset(processor, tokenizer, args.max_seq_length, "dev")

            # define the weights for each dataset loss
            split_features_dict = defaultdict(list)
            for feature in train_dataset:
                split_features_dict[feature.data_set].append(feature)
            dataset_weights = {key: 0 for key in split_features_dict.keys()}

            for dataset, features_list in split_features_dict.items():
                if args.dataset_loss_method == "weighted":
                    dataset_weights[dataset] = (1 / len(split_features_dict.keys())) \
                                               * len(features_list) / len(train_dataset)
                else:
                    dataset_weights[dataset] = (1 / len(split_features_dict.keys()))

            # initialize trainer class
            trainer = Trainer(model, args, train_dataset, validation_dataset, dataset_weights,
                              save_or_load_model_path, Data_Collator)

            # do training
            trainer.train_data(device=device, batch_size=args.batch_size)

        # start evaluation
        if args.do_eval:

            # prepare the test DataLoader
            test_dataset = convert_features_to_dataset(processor, tokenizer, args.max_seq_length, "test")
            test_data_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                collate_fn=Data_Collator.collate_batch,
            )

            # load the trained model from the provided path
            out_model = BertForSequenceClassification.from_pretrained(save_or_load_model_path,
                                                                      output_hidden_states=True)
            out_model.to(device)
            # Put model to eval mode
            out_model.eval()

            # Tracking results in a dict for each dataset since each batch might contain data from multiple data sets
            data_set_test_result_dict = {}

            # do evaluation
            for batch in test_data_loader:

                batch.move_tensors_to_device(device)

                with torch.no_grad():

                    outputs = out_model(batch, calculate_loss=False)
                # Results from the forward pass are automatically separated by data set
                for single_dataset_output in outputs.bert_single_dataset_outputs:
                    # Move predictions and labels to CPU
                    cpu_logits_lists = [slog.detach().cpu().numpy() for slog in single_dataset_output.logits]
                    cpu_labels_lists = [slab.to('cpu').numpy() for slab in single_dataset_output.labels]
                    data_set = single_dataset_output.data_set

                    # Add results to the dict
                    if data_set not in data_set_test_result_dict:
                        data_set_test_result_dict[data_set] = TestResult(data_set, cpu_logits_lists,
                                                                         cpu_labels_lists)
                    else:
                        data_set_test_result_dict[data_set].predictions += cpu_logits_lists
                        data_set_test_result_dict[data_set].true_labels += cpu_labels_lists

            # regression summary metrics
            regression_results = []

            for data_set_name, data_set_test_result in data_set_test_result_dict.items():

                data_set_size = len(data_set_test_result.true_labels)

                # Combine the correct labels for each batch into a single list.
                flat_true_labels = [label.flatten() for label in data_set_test_result.true_labels]
                flat_true_labels = np.concatenate(flat_true_labels, axis=0)

                flat_predictions = [pred.flatten() for pred in data_set_test_result.predictions]
                flat_predictions = np.concatenate(flat_predictions, axis=0)

                pearson, spearman, corr = pearson_and_spearman(flat_predictions, flat_true_labels)

                regression_result = [seed, args.dataset_loss_method, data_set_name, pearson, spearman,
                                     corr, data_set_size]

                regression_results.append(regression_result)

            reg_results_df = pd.DataFrame(regression_results)
            if not os.path.exists(args.save_results_path):
                reg_results_df.to_csv(args.save_results_path, sep=",",
                                      header=["seed", "loss_method", "dataset", "pearson", "spearman",
                                              "corr", "dataset_size"], index=False)
            else:
                reg_results_df.to_csv(args.save_results_path, index=False, mode='a', header=False)














