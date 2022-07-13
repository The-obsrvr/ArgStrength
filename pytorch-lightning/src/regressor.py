# -*- coding utf-8 -*-
"""
Defines the Model class as per the pytorch lightning structure and functions.
"""
# Standard Imports
import logging as log
from abc import ABC
from argparse import ArgumentParser
import itertools
from collections import OrderedDict

# Third party Imports
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import transformers
from transformers import (get_linear_schedule_with_warmup, AdamW, BertModel)
from torchnlp.utils import collate_tensors
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

# Local Imports
from utils import generate_task_list
from datasets import ArgTokenizer, ArgumentDataset


class ArgStrRanker(pl.LightningModule, ABC):
    """
    Class definition for the argument strength model based on the chosen hyperparameter setting. 
    """

    @property
    def hparams(self):
        return self._hparams

    class DataModule(pl.LightningDataModule, ABC):

        @property
        def hparams(self):
            return self._hparams

        def __init__(self, ranker_instance):
            super().__init__()

            self.tokenizer = ranker_instance.tokenizer
            self.hparams = ranker_instance.hparams
            self.ranker = ranker_instance
            self.weights = {}
            self.train_dataset = ArgumentDataset(self.hparams,
                                                 mode="train",
                                                 tokenizer=self.tokenizer,
                                                 )

        def train_dataloader(self) -> DataLoader:
            """ returns the train dataloader  """
            self.train_dataset = ArgumentDataset(self.hparams,
                                                 mode="train",
                                                 tokenizer=self.tokenizer,
                                                 return_weights=True
                                                 )
            self.weights = self.train_dataset.weights
            return DataLoader(
                dataset=self.train_dataset,
                sampler=RandomSampler(self.train_dataset),
                batch_size=self.hparams.train_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def val_dataloader(self) -> DataLoader:
            """ returns the validation dataloader  """
            val_dataset = ArgumentDataset(self.hparams,
                                          mode="dev",
                                          tokenizer=self.tokenizer,
                                          )
            return DataLoader(
                dataset=val_dataset,
                sampler=None,
                batch_size=self.hparams.eval_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def test_dataloader(self) -> DataLoader:
            """ returns the test dataloader  """
            test_dataset = ArgumentDataset(self.hparams,
                                           mode="test",
                                           tokenizer=self.tokenizer,
                                           )
            return DataLoader(
                dataset=test_dataset,
                sampler=None,
                batch_size=self.hparams.eval_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def predict_dataloader(self) -> DataLoader:
            """ returns the predict dataloader """
            predict_dataset = ArgumentDataset(self.hparams,
                                              mode="predict",
                                              tokenizer=self.tokenizer,
                                              )
            return DataLoader(
                dataset=predict_dataset,
                sampler=None,
                batch_size=self.hparams.eval_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        @hparams.setter
        def hparams(self, value):
            self._hparams = value

    def __init__(self, hparams) -> None:
        """
        Initialize the class object.
        :param hparams: ArgumentParser containing the hyperparameters
        """
        super(ArgStrRanker, self).__init__()

        self.hparams = hparams

        self.task_list = generate_task_list(self.hparams)

        # build model
        self._build_model()

        # Build DataModule
        self.data = self.DataModule(self)

        # Loss Criterion
        self._build_loss()

        self.nr_frozen_epochs = hparams.nr_frozen_epochs
        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.save_hyperparameters(hparams)

    def _build_model(self):
        """ Initialize Encoder model + Tokenizer + Regression Unit """
        # BERT
        self.bert = BertModel.from_pretrained(self.hparams.encoder_model,
                                              output_hidden_states=True)
        self.bert.gradient_checkpointing_enable()

        # Tokenizer
        self.tokenizer = ArgTokenizer(self.hparams.encoder_model)
        transformers.logging.set_verbosity_error()

        # Regression Head
        self.regressors = nn.ModuleDict()
        for dataset in self.task_list:
            # Multiple heads in MTL
            if "MTLAS" in self.hparams.task_name:
                self.regressors[dataset] = self._generate_regression_unit()
                continue
            #  Single head in STL
            else:
                self.regressors["regressor"] = self._generate_regression_unit()
                break

    def _generate_regression_unit(self):
        """
        define the regression unit based on the hyperparameter combination. 
        :return:
        """
        last_dim = self.bert.config.hidden_size * self.hparams.bert_hidden_layers
        layers = []
        if self.hparams.mlp_config == 1:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            last_dim = 512
        elif self.hparams.mlp_config == 2:
            layers.append(nn.Linear(in_features=last_dim, out_features=100, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=100))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            last_dim = 100
        elif self.hparams.mlp_config == 3:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            layers.append(nn.Linear(in_features=512, out_features=100, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=100))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            last_dim = 100
        else:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            layers.append(nn.Linear(in_features=512, out_features=256, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=256))
            layers.append(nn.ReLU())
            if self.hparams.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            last_dim = 256
        layers.append(nn.Linear(in_features=last_dim, out_features=1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def get_regression_unit(self, dataset):
        """Creates or returns already created regressor for a task.

                :param dataset:
                    Dataset being processed: gretz, toledo, swanson, UKPRank, webis.

                :return: Regressor that will be used for the forward pass.
                """
        #  for MTL:
        if dataset in self.task_list and "MTLAS" in self.hparams.task_name:
            return self.regressors[dataset]
        #  for STL:
        elif self.regressors is not None and "STLAS" in self.hparams.task_name:
            return self.regressors["regressor"]
        else:
            raise ValueError("Incorrect Dataset or Task detected. Check your input")

    def _build_loss(self):
        """ Initialize loss function """
        self._loss = nn.MSELoss()

    def unfreeze_encoder(self) -> None:
        if self._frozen:
            log.info("\n-- Encoder model fine-tuning")
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """freezes the encoder layer."""
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, input_ids, attention_mask, datasets) -> dict:
        """
        Modified forward function adjusted for STL and MTL models.
        :param input_ids: [batch_size * max_length]
        :param attention_mask: [batch_size * max_length]
        :param datasets: [batch_size]
        :return:
        """

        outputs = self.bert(input_ids, attention_mask)
        # print(outputs.shape)
        hidden_states = outputs.hidden_states
        out = torch.cat(tuple([hidden_states[-i]
                               for i in range(1, self.hparams.bert_hidden_layers + 1)]),
                        dim=-1)
        # print(out.shape)
        # Pooling by setting masked items to zero
        bert_mask = attention_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-padding tokens
        out = torch.mul(out, bert_mask)
        # Select the first token of the seq 'CLS'
        out = out[:, 0, :]
        # print(out.shape)

        if "unk" not in datasets:
            # source of argument is known. Respective regression unit can be used to
            # make prediction.
            # Issue: even if one data source is unknown, all arguments are sent for the alternative method.

            logits = []
            for i, dataset in enumerate(datasets):
                regressor = self.get_regression_unit(dataset)
                logits_by_dataset = regressor(out[i, :])
                logits.extend(logits_by_dataset)

            assert len(logits) == len(datasets)
            # convert to a single tensor containing the logits of the batch.
            logits = torch.stack(logits)
            # print(logits)
            return {"logits": logits}

        else:
            # source of argument is unknown. All reg units are used to generate the prediction
            # and an aggregation may follow.
            logits_dict = {}
            for dataset_reg_unit in self.task_list:
                regressor = self.get_regression_unit(dataset_reg_unit)
                logits_dict[dataset_reg_unit] = regressor(out[:])

            return logits_dict

    def loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """

        :param predictions: dict containing the predicted logits and the datasets info.
        :param targets: dict containing the label/target value.
        :return:
        """
        if self.hparams.task_name:
            # only single dataset is in the training dataset and loss doesn't require aggregation.
            return self._loss(predictions["logits"], targets["labels"])
        else:
            losses = {}
            # 1. make a dataframe
            output_data = {"logits": predictions["logits"].tolist(),
                           "datasets": predictions["datasets"],
                           "labels": targets["labels"].tolist()}
            output_df = pd.DataFrame(output_data)
            # 2. group by dataset
            grouped_output = output_df.groupby("datasets")
            # 3. loss by dataset
            for dataset in grouped_output.groups.keys():
                group_by_dataset = grouped_output.get_group(dataset)
                logits = torch.tensor(group_by_dataset["logits"].values)
                labels = torch.tensor(group_by_dataset["labels"].values)
                losses[dataset] = self._loss(logits, labels)

            # 4. aggregate the loss
            if self.hparams.dataset_loss_method == "unweighted":
                total_loss = sum(losses.values())

            else:
                total_loss_list = {k: v.item() * self.data.weights[k]
                                   for k, v in losses.items()
                                   if k in self.data.weights}
                total_loss = sum(total_loss_list.values())
            # print(total_loss)
            return torch.tensor(total_loss, dtype=torch.float, requires_grad=True)

    def cal_performance(self, predictions: dict,
                        targets: dict,
                        return_by_dataset: bool = False,
                        log_metrics: bool = True):
        """
        Calculates the performance metrics: Pearson's Correlation Coefficient and Spearman Rank Correlation Coefficient by dataset and overall.
        :param log_metrics: Whether to log the metrics values to MLFlow.
        :param return_by_dataset: Whether to return the performance metrics by dataset value.
        :param predictions: Dict containing: logits: single tensor containing logit values; datasets
        :param targets: dict containing labels
        :return:
        """
        pearson_corr = {}
        spearman_corr = {}
        # 1. make a dataframe of the logits
        output_data = {"logits": predictions["logits"],
                       "datasets": predictions["datasets"],
                       "labels": targets["labels"]}
        output_df = pd.DataFrame(output_data)
        # print(output_df.head(10))
        logits = torch.tensor(output_df["logits"].values)
        labels = torch.tensor(output_df["labels"].values)
        # print(logits)
        # print(labels)
        pearson_corr_overall = pearson_corrcoef(logits, labels)

        # 2. group by dataset
        grouped_output = output_df.groupby("datasets")
        # 3. metrics by dataset
        for dataset in grouped_output.groups.keys():
            group_by_dataset = grouped_output.get_group(dataset)
            logits = torch.tensor(group_by_dataset["logits"].values)
            labels = torch.tensor(group_by_dataset["labels"].values)
            pearson_corr[dataset] = pearson_corrcoef(logits, labels)
            spearman_corr[dataset] = spearman_corrcoef(logits, labels)

            # log the pearson and spearman values by dataset.
            if log_metrics:
                self.log(dataset + "_pearson", pearson_corr[dataset].item())
                self.log(dataset + "_spearman", spearman_corr[dataset].item())

        if return_by_dataset:
            return pearson_corr_overall, pearson_corr, spearman_corr
        else:
            return pearson_corr_overall

    def training_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """
        Run one training step consisting of one forward function and one loss function
        :param batch: batch containing the data inputs as defined by the dataloader and Dataset class object.
        :param batch_idx: batch id

        :return: output from the model. 
        """
        inputs, targets = batch
        assert len(inputs["input_ids"]) == len(targets["labels"])
        model_out = self.forward(**inputs)

        loss_value = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # loss_value = loss_value.unsqueeze(0)
        # print(loss_value)

        self.log("train_loss", loss_value, on_epoch=True, batch_size=self.hparams.train_batch_size)
        tqdm_dict = {"train_loss": loss_value.detach()}
        output = OrderedDict(
            {"loss": loss_value, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """
        Run one validation step consisting of one forward function and one loss function
        :param batch: batch containing the data inputs as defined by the dataloader and Dataset class object.
        :param batch_idx: batch id

        :return: output from the model.
        """
        inputs, targets = batch
        assert len(inputs["input_ids"]) == len(targets["labels"])

        model_out = self.forward(**inputs)
        loss_value = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # loss_value = loss_value.unsqueeze(0)

        tqdm_dict = {"val_loss": loss_value.detach()}
        output = OrderedDict(
            {
                "dataset": inputs["datasets"],
                "logits": model_out["logits"].cpu().numpy(),
                "labels": targets["labels"].cpu().numpy(),
                "val_loss": loss_value,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict
            }
        )
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs: dict) -> dict:
        """Function that takes as input a list of dictionaries returned by the validation_step function and measures the model performance across the entire validation set.
                Returns:
                    - Dictionary with metrics to be added to the lightning logger.
                """
        #  Calculate the correlation coefficients.
        d = {}
        for k in ["logits", "labels", "dataset"]:
            d[k] = [output[k] for output in outputs]
        predictions = {"logits": np.concatenate(d["logits"]).ravel(),
                       "datasets": list(itertools.chain.from_iterable(d["dataset"]))}
        targets = {"labels": np.concatenate(d["labels"]).ravel()}

        pearson_corr_overall = self.cal_performance(predictions, targets)
        self.log("val_pearson_coeff", pearson_corr_overall.item(),
                 on_epoch=True,
                 batch_size=self.hparams.eval_batch_size)

        #  Calculate the validation loss
        val_loss_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]
            # reduce manually when using dp
            val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
        val_loss_mean /= len(outputs)
        self.log("val_loss", val_loss_mean, on_epoch=True, batch_size=self.hparams.eval_batch_size)

        tqdm_dict = {"val_loss": val_loss_mean,
                     "val_pearson_coeff": pearson_corr_overall}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "val_pearson_coeff": pearson_corr_overall
        }
        return result

    def test_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """
        Runs single testing step consisting of one forward function and one loss function. 
        :param batch: batch
        :param batch_idx: batch id
        :return: test output of one batch.
        """
        inputs, targets = batch
        assert len(inputs["input_ids"]) == len(targets["labels"])

        model_out = self.forward(**inputs)

        loss_value = self.loss(model_out, targets)
        loss_value = loss_value.unsqueeze(0)

        output = OrderedDict(
            {
                "dataset": inputs["datasets"],
                "logits": model_out["logits"].cpu().numpy(),
                "labels": targets["labels"].cpu().numpy(),
                "test_loss": loss_value,
            }
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the validation_step function and measures the model performance across the entire validation set.
                        Returns:
                            - Dictionary with metrics to be added to the lightning logger.
                        """

        d = {}
        for k in ["logits", "labels", "dataset"]:
            d[k] = [output[k] for output in outputs]
        predictions = {"logits": np.concatenate(d["logits"]).ravel(),
                       "datasets": list(itertools.chain.from_iterable(d["dataset"]))}
        targets = {"labels": np.concatenate(d["labels"]).ravel()}
        pearson_corr_overall, pearson_corr, spearman_corr = self.cal_performance(predictions,
                                                                                 targets,
                                                                                 return_by_dataset=True)
        self.log("test_pearson_coeff", pearson_corr_overall.item(), on_epoch=True)
        test_loss_mean = 0
        for output in outputs:
            test_loss = output["test_loss"]
            # reduce manually when using dp
            test_loss = torch.mean(test_loss)
            test_loss_mean += test_loss
        test_loss_mean /= len(outputs)
        self.log("test_loss", test_loss_mean, on_epoch=True)

        result = {
            "test_pearson_coeff": pearson_corr_overall,
            "pearson_by_dataset": pearson_corr,
            "spearman_by_dataset": spearman_corr
        }
        return result

    def predict(self, batch, device) -> dict:
        """
        Performs prediction step for one batch input.
        :param device: device on which the computation is performed.
        :param batch: ith predict batch
        :return: logits_dict
        """
        inputs, _ = batch
        input_ids = inputs["input_ids"].clone().detach().to(device)
        attention_masks = inputs["attention_mask"].clone().detach().to(device)
        logits_dict = self.forward(input_ids, attention_masks, inputs["datasets"])

        return logits_dict

    @staticmethod
    def prepare_sample(sample: list) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        inputs = {"input_ids": sample["input_ids"],
                  "attention_mask": sample["attention_mask"],
                  "datasets": sample["dataset"]}

        targets = {"labels": sample["label"]}

        if None not in targets.values():
            return inputs, targets

        else:
            print("Label or labels are not available.")
            return inputs, {}

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [
            # regression unit parameters
            {
                "params": self.regressors.parameters(),
                "weight_decay": self.hparams.weight_decay
            },
            #  BERT parameters.
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        total_steps = len(self.data.train_dataset) * self.hparams.max_epochs
        # Learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=total_steps)

        return [optimizer], [scheduler]

    def on_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.hparams.nr_frozen_epochs:
            self.unfreeze_encoder()

    @staticmethod
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    @hparams.setter
    def hparams(self, value):
        self._hparams = value

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """

        :param parser:
        :return: updated parser
        """
        #  Model Arguments
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=2,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--bert_hidden_layers",
            default=4,
            type=int,
            help="Number of bert hidden layers to concat and feed into the \
                          regression unit. Options: '1, 2, 3, 4'",
        )
        parser.add_argument(
            "--dropout_prob",
            default=0.2,
            type=float,
            help="dropout probability to use. Default: '0.2'",
        )
        parser.add_argument(
            "--mlp_config",
            default=2,
            type=int,
            help="Neural Net Architecture of the Regression Unit. Options: Arch '1,2,3,4'",
        )

        #  Optimizer and Scheduler
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-06,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-06,
            type=float,
            help="Regression Unit learning rate.",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.07,
            type=float,
            help="Regression Unit weight decay. Default: '0.07'",
        )
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer. Default: '1e-8'",
        )
        parser.add_argument(
            "--warmup_steps",
            default=2,
            type=int,
            help="Warmup steps for scheduler. Default: '2'",
        )

        #  Data Related Arguments
        parser.add_argument(
            "--loader_workers",
            default=16,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                        the data will be loaded in the main process.",
        )
        parser.add_argument(
            "--max_seq_length",
            default=None,
            type=int,
            help="maximum sequence length to be used for encoding by Bert Tokenizer. Default: 'None' \
            (model finds max_length from data)",
        )
        parser.add_argument(
            "--max_seq_length_perc",
            default=99.5,
            type=float,
            help="maximum sequence length percentile to be used for finding max seq length from data. Default: 99 \
                    (model finds max_length from data)",
        )
        parser.add_argument(
            "--dataset_loss_method",
            default="unweighted",
            type=str,
            help="Whether to use weighted (by dataset) loss or not. Default: unweighted",
        )
        parser.add_argument(
            "--train_batch_size", default=64, type=int, help="Train Batch size to be used."
        )
        parser.add_argument(
            "--eval_batch_size", default=64, type=int, help="Eval Batch size to be used."
        )
        parser.add_argument(
            "--sampling_strategy",
            default="cross-topic",
            choices=["in-topic", "cross-topic", "balanced"],
            type=str,
            help="Sampling Strategy to use. Default: cross-topic"
        )
        return parser
