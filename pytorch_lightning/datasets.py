# -*- coding: utf-8 -*-

import os
import glob
import random

# third party imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from .utils import generate_task_dict


class ArgTokenizer:
    """

    """
    def __init__(self, pretrained_model):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def batch_encode(self,
                     sentence: str,
                     sentence_pair: str = None,
                     max_length: int = 128) -> (torch.tensor, torch.tensor):
        """
        :param sentence:
        :param max_length:
        :param sentence_pair:
        :return:
        """
        tokenizer_output = self.tokenizer.encode_plus(
            text=sentence,
            text_pair=sentence_pair,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return tokenizer_output


class ArgumentDataset(Dataset):
    """

    """
    def __init__(self,
                 hparams,
                 mode: str,
                 tokenizer: ArgTokenizer,
                 return_weights: bool = False
                 ):
        """

        :param hparams:
        :param mode: train/dev/test/predict set to produce
        :param tokenizer:
        """
        super().__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        if self.hparams.max_seq_length is None:
            self.max_length = self._find_max_length()
        else:
            self.max_length = self.hparams.max_seq_length

        self.data = self._load_data_files(self.hparams.data_folder_path, mode)

        if return_weights:
            self.weights = self._get_weights_by_datasets()

    def _load_data_files(self, data_folder_path: str, mode: str) -> pd.DataFrame:
        """

        :param data_folder_path:
        :param mode:
        :return:
        """
        datasets = glob.glob(data_folder_path)
        task_dict = generate_task_dict(self.hparams)
        data_df = pd.DataFrame()

        # iterate across each data file and add it to the dataframe.
        for i, dataset_path in enumerate(datasets):
            dataset_name = os.path.basename(dataset_path)[:-4]
            # only read datasets based on the defined task [filtering]
            if dataset_name in task_dict:
                loaded_data = pd.read_csv(dataset_path)
                loaded_data["dataset"] = dataset_name
                if "T_dist" in self.hparams.task_name:
                    if dataset_name == "toledo":
                        loaded_data["modified_topic"] = loaded_data["topic"].str[:-6]
                    elif dataset_name == "ukp":
                        loaded_data["modified_topic"] = loaded_data["topic"].str.split("_", n=1, expand=True)[0]
                    else:
                        loaded_data["modified_topic"] = loaded_data["topic"]
                    loaded_data = self._topic_splitter(loaded_data, dataset_name)
                data_df = data_df.append(loaded_data, ignore_index=True)

        # select rows corresponding to the set [splitting]
        data_df = data_df[data_df["set"] == mode]
        return data_df

    @staticmethod
    def _topic_splitter(data_df, dataset_name) -> pd.DataFrame:
        """
            Splits the datasets into the train-dev-test sets based on the topic distribution.
            :return:
            """
        ds_topic_list = data_df["modified_topic"].unique()
        random.shuffle(ds_topic_list)
        if dataset_name == "swanson":
            ratio = 1
        else:
            ratio = int(len(ds_topic_list) / 5)

        ds_topic_test = ds_topic_list[:ratio]
        ds_topic_dev = ds_topic_list[ratio:2 * ratio]
        ds_topic_train = ds_topic_list[2 * ratio:]

        data_df.loc[data_df["modified_topic"].isin(ds_topic_train), "set"] = "train"
        data_df.loc[data_df["modified_topic"].isin(ds_topic_dev), "set"] = "dev"
        data_df.loc[data_df["modified_topic"].isin(ds_topic_test), "set"] = "test"

        return data_df

    def _find_max_length(self) -> int:
        """
        find the 99th percentile length of the arguments to be used as the max length
        :return:
        """
        self.data["arg_length"] = self.data["argument"].apply(lambda x: len(x.split()))
        return int(np.percentile(self.data["arg_length"], 99))

    def _get_weights_by_datasets(self) -> dict:
        """

        :return:
        """
        dataset_groups = self.data.groupby("dataset")
        dataset_weights = {}
        for dataset in dataset_groups.groups.keys():
            if self.hparams.dataset_weight_method == "unweighted":
                dataset_weights[dataset] = 1
            # else dataset weight method is weighted
            else:
                dataset_weights[dataset] = (len(self.data)) / (len(dataset_groups.groups[dataset]))
        return dataset_weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        arg = data_row.argument
        label = data_row.label
        topic = data_row.modified_topic
        data_set = data_row.dataset
        if "source" in self.hparams.task_name:
            text_b = data_set
        elif "topic" in self.hparams.task_name:
            text_b = topic
        else:
            text_b = None

        encoding = self.tokenizer.batch_encode(arg, text_b, self.max_length)

        return dict(
            arg_text=arg,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=torch.tensor(label, dtype=torch.float),
            dataset=data_set
        )


# :TODO: define changes for "predict" mode. - only contains text