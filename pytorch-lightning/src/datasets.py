# -*- coding: utf-8 -*-

import os
import glob
import random

# third party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import generate_task_list


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
            truncation='only_first',
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
                 data_folder: str = None,
                 source_known: bool = True,
                 prepare_target: bool = True,
                 return_weights: bool = False,
                 return_files: bool = False
                 ):
        """

        :param hparams:
        :param mode: train/dev/test/predict set to produce
        :param tokenizer:
        """
        super().__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        self.mode = mode
        self.source_known = source_known
        self.prepare_target = prepare_target

        self.data_folder = self.hparams.data_folder_path\
            if data_folder is None\
            else data_folder

        self.data = self._load_data_files()

        if return_weights:
            self.weights = self._get_weights_by_datasets()
        if return_files:
            self.file_names = self._get_file_names()

    def _get_file_names(self):
        """

        :return:
        """
        datasets = glob.glob(self.data_folder)
        return [os.path.basename(dataset_path)[:-4] for dataset_path in datasets]

    def _load_data_files(self) -> pd.DataFrame:
        """
        Reads data files as per the task and sampling strategy (default: "IN-TOPIC")
        :return:
        """
        datasets = glob.glob(self.data_folder)
        if self.hparams.sampling_strategy == "balanced":
            smallest_dataset_name, smallest_dataset_len = self._find_smallest_set(datasets)
        else:
            smallest_dataset_name, smallest_dataset_len = None, None

        train_task_list = generate_task_list(self.hparams)

        if self.mode == "train":
            print("Datasets used to train on: ", train_task_list)
            print("Sampling Strategy: ", self.hparams.sampling_strategy)

        data_df = pd.DataFrame()

        # iterate across each data file and add it to the dataframe if included in task.
        for i, dataset_path in enumerate(datasets):
            dataset_name = os.path.basename(dataset_path)[:-4]

            # only read datasets specific to the selected task [filtering]
            if self.mode == "train" or self.mode == "dev"\
                    or (self.mode == "test" and "MTLAS_LOO" in self.hparams.task_name):
                if dataset_name in train_task_list:
                    loaded_data = pd.read_csv(dataset_path)
                    loaded_data["dataset"] = dataset_name

                    #  Split data if sampling strategy " CROSS-TOPIC "
                    if self.hparams.sampling_strategy == "cross-topic":
                        ct_data = self._split_by_cross_topic(loaded_data)
                        data_df = pd.concat([ct_data, data_df], ignore_index=True)

                    elif self.hparams.sampling_strategy == "balanced":
                        bal_data = self._split_by_equal_rep(loaded_data, smallest_dataset_name, smallest_dataset_len)
                        data_df = pd.concat([bal_data, data_df], ignore_index=True)

                    else:
                        # sampling strategy is in-topic. Loaded data is in in-topic format already.
                        data_df = pd.concat([loaded_data, data_df], ignore_index=True)

            # no filtering by task name required in testing and predict mode.
            elif self.mode == "test" or self.mode == "predict":
                loaded_data = pd.read_csv(dataset_path)
                loaded_data["dataset"] = dataset_name

                #  Split data if sampling strategy " CROSS-TOPIC "
                if self.hparams.sampling_strategy == "cross-topic":
                    loaded_data = self._split_by_cross_topic(loaded_data)
                    data_df = pd.concat([data_df, loaded_data], ignore_index=True)

                elif self.hparams.sampling_strategy == "balanced":
                    loaded_data = self._split_by_equal_rep(loaded_data, smallest_dataset_name, smallest_dataset_len)
                    data_df = pd.concat([data_df, loaded_data], ignore_index=True)

                else:
                    data_df = pd.concat([data_df, loaded_data], ignore_index=True)

            else:
                raise ValueError("Dataset path may be incorrect. Please check the data folder.")

        #  define the max seq length based on the complete data (based on training task)
        #  if hparams doesn't have a max_seq_length value
        if self.hparams.max_seq_length is None:
            self.hparams.max_seq_length = self._find_max_length(data_df)

        # print(len(data_df))

        if self.mode == "predict":
            #  reset predict dataset to test set as they are the same in our current setup.
            self.mode = "test"

        # Split data if sampling strategy " EQUAL-REP "
        if self.hparams.sampling_strategy == "equal-rep":
            #  split train-dev-test
            bal_training, bal_test = train_test_split(data_df, test_size=0.15, random_state=42, shuffle=True)
            bal_train, bal_val = train_test_split(bal_training, test_size=0.10, random_state=42, shuffle=True)
            if self.mode == "train":
                return bal_train
            elif self.mode == "dev":
                return bal_val
            else:
                return bal_test
        else:
            # select rows corresponding to the set [splitting]
            data_df = data_df[data_df["set"] == self.mode]
            return data_df

    def _split_by_cross_topic(self, data_df) -> pd.DataFrame:
        """
            Splits the datasets into the train-dev-test single set based on the sampling strategy " CROSS-TOPIC "
            :return:
            """
        ds_topic_list = data_df["topic"].unique()
        random.Random(self.hparams.run_seed).shuffle(ds_topic_list)
        if data_df["dataset"].any() == "swanson":
            ratio = 1
        else:
            ratio = int(len(ds_topic_list) / 5)

        ds_topic_test = ds_topic_list[:ratio]
        ds_topic_dev = ds_topic_list[ratio:2 * ratio]
        ds_topic_train = ds_topic_list[2 * ratio:]

        data_df.loc[data_df["topic"].isin(ds_topic_train), "set"] = "train"
        data_df.loc[data_df["topic"].isin(ds_topic_dev), "set"] = "dev"
        data_df.loc[data_df["topic"].isin(ds_topic_test), "set"] = "test"

        return data_df

    def _split_by_equal_rep(self, data_df: pd.DataFrame, smallest_dataset_name: str, small_dataset_len: int)\
            -> pd.DataFrame:
        """
        Splits the dataset based on the sampling strategy " Equal Representation "
        :param data_df:
        :return:
        """
        if data_df["dataset"].str.contains(smallest_dataset_name).any():
            #  no filtering required for smallest dataset. Read as is.
            return data_df
        else:
            # number of samples to be drawn per topic
            samples_to_be_drawn = int(np.ceil(small_dataset_len / len(data_df["topic"].unique())))
            # print("Sample drawn from {} per topic: {}".format(data_df["dataset"][0], samples_to_be_drawn))

            dataset_topic_grouped = data_df.groupby("topic")
            sample_df = pd.DataFrame()
            for topic, topic_grp in dataset_topic_grouped:
                # samples per topic
                samples = topic_grp.sample(n=samples_to_be_drawn, replace=False, random_state=self.hparams.run_seed)
                sample_df = pd.concat([sample_df, samples])

        return sample_df

    def _find_max_length(self, data: pd.DataFrame) -> int:
        """
        find the pth percentile length of the arguments to be used as the max length
        :return:
        """
        sens = list(data["argument"])
        topics = list(data["topic"])
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if self.hparams.use_topic:
            batch_encoding = tokenizer.batch_encode_plus(
                [(sen, topic) for (sen, topic) in zip(sens, topics)], pad_to_max_length=False,
                )
        else:
            batch_encoding = tokenizer.batch_encode_plus(
                [sen for sen in sens], pad_to_max_length=False,
                )

        # Collect all lengths
        data["arg_length"] = [len(x) for x in batch_encoding['input_ids']]
        return int(np.percentile(data["arg_length"], self.hparams.max_seq_length_perc))

    @staticmethod
    def _find_smallest_set(data_sets: list):
        """

        :param data_sets:
        :return:
        """
        data_sizes = {}
        for data_path in data_sets:
            data_df = pd.read_csv(data_path)
            data_name = os.path.basename(data_path)[:-4]
            data_sizes[data_name] = len(data_df)
        # return smallest data size
        return min(data_sizes, key=data_sizes.get), data_sizes[min(data_sizes, key=data_sizes.get)]

    def _get_weights_by_datasets(self) -> dict:
        """

        :return:
        """
        dataset_groups = self.data.groupby("dataset")
        dataset_weights = {}
        for dataset in dataset_groups.groups.keys():
            if self.hparams.dataset_loss_method == "unweighted":
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

        if self.prepare_target:
            label = torch.tensor(data_row.label, dtype=torch.float)
        else:
            label = None

        if self.source_known is False:
            data_set = "unk"
        else:
            data_set = data_row.dataset

        topic = data_row.topic
        if "source" in self.hparams.task_name:
            text_b = data_set
        elif self.hparams.use_topic:
            text_b = topic
        else:
            text_b = None

        encoding = self.tokenizer.batch_encode(arg, text_b, self.hparams.max_seq_length)

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=label,
            dataset=data_set
        )
