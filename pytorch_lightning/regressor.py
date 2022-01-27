# -*- coding utf-8 -*-

# Standard Imports
import logging as log
from abc import ABC
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
# Third party Imports
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import (get_linear_schedule_with_warmup, AdamW, BertModel)
from torchnlp.utils import collate_tensors
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
# Local Imports
from .utils import *
from .datasets import ArgTokenizer, ArgumentDataset


class ArgStrRanker(pl.LightningModule, ABC):
    """

    """

    class DataModule(pl.LightningDataModule, ABC):
        def __init__(self, ranker_instance):
            super().__init__()

            self.tokenizer = ranker_instance.tokenizer
            self.hparams = ranker_instance.hparams
            self.ranker = ranker_instance
            self.weights = {}

        def train_dataloader(self) -> DataLoader:
            """

            :return:
            """
            self._train_dataset = ArgumentDataset(self.hparams,
                                                  mode="train",
                                                  tokenizer=self.tokenizer,
                                                  return_weights=True
                                                  )
            self.weights = self._train_dataset.weights

            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.train_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def val_dataloader(self) -> DataLoader:
            """

            :return:
            """
            self._val_dataset = ArgumentDataset(self.hparams,
                                                mode="dev",
                                                tokenizer=self.tokenizer,
                                                )

            return DataLoader(
                dataset=self._val_dataset,
                sampler=None,
                batch_size=self.hparams.eval_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def test_dataloader(self) -> DataLoader:
            """

            :return:
            """
            self._test_dataset = ArgumentDataset(self.hparams,
                                                 mode="test",
                                                 tokenizer=self.tokenizer,
                                                 )

            return DataLoader(
                dataset=self._test_dataset,
                sampler=None,
                batch_size=self.hparams.eval_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

        def predict_dataloader(self) -> DataLoader:
            """

            :return:
            """
            self._predict_dataset = ArgumentDataset(self.hparams,
                                                  mode="predict",
                                                  tokenizer=self.tokenizer,
                                                  )
            return DataLoader(
                dataset=self._predict_dataset,
                sampler=RandomSampler(self._predict_dataset),
                batch_size=self.hparams.train_batch_size,
                collate_fn=self.ranker.prepare_sample,
                num_workers=self.hparams.loader_workers
            )

    def __init__(self, hparams: Namespace) -> None:
        """

        :param hparams: ArgumentParser containing the hyperparameters
        """
        super(ArgStrRanker, self).__init__()

        self.hparams = hparams
        self.task_dict = generate_task_dict(self.hparams)

        # build model
        self._build_model()

        # Build DataModule
        self.data = self.DataModule(self)

        # Loss Criterion
        self._build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

    def _build_model(self):
        """ Initialize BERT model + Tokenizer + Regression Unit """
        # BERT
        self.bert = BertModel.from_pretrained(self.hparams.encoder_model,
                                              output_hidden_states=True)
        # Tokenizer
        self.tokenizer = ArgTokenizer(self.hparams.encoder_model)

        # Regression Head
        if "MTLAS" in self.hparams.task_name:
            self.regressors = nn.ModuleDict()
            for dataset in self.task_dict:
                self.regressors[dataset] = self.generate_regression_unit(dataset)
        else:
            self.regressor = self.generate_regression_unit()

    def _generate_regression(self):
        """

        :return:
        """
        last_dim = self.bert.hidden_size * self.hparams.bert_hidden_layers
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

    def get_or_create_regression(self, dataset):
        """Creates or returns already created regressor for a task.

                :param dataset:
                    Dataset being processed: gretz, toledo, swanson, UKPRank.

                :return: Regressor that will be used for the forward pass.
                """
        if self.regressors[dataset] is not None:
            return self.regressors[dataset]
        else:
            self.regressors[dataset] = self.generate_regressors()
            return self.regressors[dataset]

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
        hidden_states = outputs.hidden_states
        out = torch.cat(tuple([hidden_states[-i]
                               for i in range(1, self.hparams.bert_hidden_layers+1)]),
                        dim=-1)

        # Pooling by setting masked items to zero
        bert_mask = attention_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-padding tokens
        out = torch.mul(out, bert_mask)
        # Select the first token of the seq 'CLS'
        out = out[:, 0, :]

        logits = []
        loss = 0
        if "MTLAS" in self.hparams.task_name:
            for i, dataset in enumerate(datasets):
                regressor = self.get_or_create_regression(dataset)
                logits.append(regressor(out[i, :]))
        else:
            logits = self.regressor(out)

        return {"logits": logits, "datasets": datasets}

    def loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """

        :param predictions:
        :param targets:
        :return:
        """
        if "only" in self.hparams.task_name:
            return self._loss(predictions["logits"], targets["labels"])
        else:
            losses = {}
            # 1. make a dataframe
            output_data = {"logits": predictions["logits"],
                           "datasets": predictions["datasets"],
                           "labels": targets["labels"]}
            output_df = pd.DataFrame(output_data)
            # 2. group by dataset
            grouped_output = output_df.groupby("datasets")
            # 3. loss by dataset
            for dataset in grouped_output.groups.keys():
                group_by_dataset = grouped_output.get_group(dataset)
                losses[dataset] = self._loss(group_by_dataset["logits"], group_by_dataset["labels"])

            # 4. aggregate the loss
            if self.hparams.dataset_loss_method is "unweighted":
                total_loss = sum(losses.values())
            else:
                total_loss_list = {k: v * self.data.weights[k]
                                   for k, v in losses.items()
                                   if k in self.data.weights}
                total_loss = sum(total_loss_list.values())

            return total_loss

    def cal_performance(self, predictions: dict, targets: dict):
        """

        :param predictions:
        :param targets:
        :return:
        """
        pearson_corr = {}
        spearman_corr = {}
        # 1. make a dataframe
        output_data = {"logits": predictions["logits"],
                       "datasets": predictions["datasets"],
                       "labels": targets["labels"]}
        output_df = pd.DataFrame(output_data)
        pearson_corr_overall = pearson_corrcoef(output_df["logits"], output_df["labels"])

        # 2. group by dataset
        grouped_output = output_df.groupby("datasets")
        # 3. metrics by dataset
        for dataset in grouped_output.groups.keys():
            group_by_dataset = grouped_output.get_group(dataset)
            pearson_corr[dataset] = pearson_corrcoef(group_by_dataset["logits"], group_by_dataset["labels"])
            spearman_corr[dataset] = spearman_corrcoef(group_by_dataset["logits"], group_by_dataset["labels"])

            # log the pearson and spearman values by dataset.
            self.log(dataset, {"pearson_by_dataset": pearson_corr[dataset],
                               "spearman_by_dataset": spearman_corr[dataset]})

        return pearson_corr_overall

    def training_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """
        Run one training step consisting of one forward function and one loss function
        :param batch:
        :param batch_idx:
        :param args:
        :param kwargs:
        :return:
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_value = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_value}
        output = OrderedDict(
            {"loss": loss_value, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        """

        :param batch:
        :param batch_idx:
        :param args:
        :param kwargs:
        :return:
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_value = self.loss(model_out, targets)

        # pearson coeff.
        _ = self.cal_performance(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_value = loss_value.unsqueeze(0)

        output = OrderedDict(
            {
                "dataset": model_out["datasets"],
                "logits": model_out["logits"],
                "labels": targets["labels"],
                "val_loss": loss_value,
            }
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """Function that takes as input a list of dictionaries returned by the validation_step
                function and measures the model performance across the entire validation set.
                Returns:
                    - Dictionary with metrics to be added to the lightning logger.
                """
        val_loss_mean = 0
        d = {}
        for k in ["logits", "labels", "datasets"]:
            d[k] = tuple(output[k] for output in outputs)
        predictions = {"logits": d["logits"], "datasets": d["datasets"]}
        targets = {"labels": d["labels"]}
        pearson_corr_overall = self.cal_performance(predictions, targets)

        for output in outputs:
            val_loss = output["val_loss"]
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
        val_loss_mean /= len(outputs)

        tqdm_dict = {"val_loss": val_loss_mean, "pearson_coeff": pearson_corr_overall}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
            "pearson_coeff": pearson_corr_overall
        }
        return result


    @staticmethod
    def prepare_sample(sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param prepare_target:
        :param sample: list of dictionaries.
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        inputs = {"input_ids": sample["input_ids"],
                  "attention_mask": sample["attention_mask"],
                  "datasets": sample["dataset"]}

        if not prepare_target:
            return inputs, {}
        else:
            targets = {"labels": sample["label"]}
            return inputs, targets

    def configure_optimizers(self):
        """Sets different Learning rates for different parameter groups."""
        parameters = [
            {"params": self.regressor.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        total_steps = len(self._training_data) * self.hparams.max_num_train_epochs
        # Learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=total_steps)

        return [optimizer], [scheduler]

    def on_epoch_end(self):
        """Pytorch lightning hook"""
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """

        :param parser:
        :return: updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default="bert-base-uncased",
            type=str,
            help="Encoder model to be used.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Regression Unit learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--data_folder_path",
            default="/data/*.csv",
            type=str,
            help="Path to the file containing the data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=3,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                        the data will be loaded in the main process.",
        )
        return parser

    @staticmethod
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    def predict(self, sample: dict) -> dict:
        """

        :param sample:
        :return:
        """
        if self.training:
            self.eval()

        # apply dropout
        self.apply(self.apply_dropout)


#  :TODO: forward with inference, and predict functions.
#  Question: Does freezing a layer work alongside warmup scheduler?
#  Question: Define separate optimizers for each head?