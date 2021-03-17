# Standard Imports
import csv
import os
import glob
from typing import List, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum


# Third Party Imports
import torch
import pandas as pd


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification.

    Attributes:
        InputExample.guid: Unique id for the example.
        InputExample.text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        InputExample.text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        InputExample.label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        InputExample.task: Task the model will run.
        InputExample.data_set: Data set the model will run
    """
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    data_set: str = None


@dataclass(frozen=True)
class InputFeatures:
    """A single set of features of data.

    Property names are the same names as the corresponding inputs to a model.
    Attributes:
        InputFeatures.input_ids: Indices of input sequence tokens in the vocabulary.
        InputFeatures.attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        InputFeatures.token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        InputFeatures.label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        InputFeatures.task: Task of the input feature.
        InputFeatures.data_set: Data set of the input feature.
        InputFeatures.label_counts: List of label counts
    """
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    data_set: str = None


@dataclass
class BertSingleDatasetInput:
    """Data class for BERT forward pass input for a single dataset

    Attributes:
        BertSingleDatasetInput.labels: Tensor of the true labels
        BertSingleDatasetInput.attention_mask: Tensor of the attention mask
        BertSingleDatasetInput.input_ids: Tensor of the tokenized input_ids
        BertSingleDatasetInput.data_set: Name of the data set of this input
    """
    labels: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    data_set: str


@dataclass
class BertSingleDatasetOutput:
    """Data class for BERT forward pass output for a single dataset

    Attributes:
        BertSingleDatasetOutput.labels: True labels
        BertSingleDatasetOutput.logits: Predicted logits
        BertSingleDatasetOutput.oss: Loss value of the prediction
        BertSingleDatasetOutput.task: Task of the prediction
        BertSingleDatasetOutput.data_set: Data set name of the prediction

    """
    labels: torch.Tensor
    logits: torch.Tensor
    loss: torch.Tensor
    data_set: str


class BertBatchInput:
    """Utility class containing everything needed for a single BERT batch input forward pass."""
    bert_single_dataset_inputs: List[BertSingleDatasetInput]

    def __init__(self, bert_single_dataset_inputs):
        self.bert_single_dataset_inputs = bert_single_dataset_inputs

    def move_tensors_to_device(self, device):
        for single_dataset_input in self.bert_single_dataset_inputs:
            single_dataset_input.labels = single_dataset_input.labels.to(device)
            single_dataset_input.attention_mask = single_dataset_input.attention_mask.to(device)
            single_dataset_input.input_ids = single_dataset_input.input_ids.to(device)


class BertBatchOutput:
    """Utility class containing output of a BERT batch forward pass"""
    bert_single_dataset_outputs: List[BertSingleDatasetOutput]

    def __init__(self, bert_single_dataset_outputs):
        self.bert_single_dataset_outputs = bert_single_dataset_outputs

    def calculate_weighted_loss(self, dataset_weights) -> float:
        """
        Calculates the weighted loss of the batch output. Weighs each dataset in the batch
        using the provided dataset_weights
        """
        loss = sum([ds.loss * dataset_weights[ds.data_set]
                    for ds in self.bert_single_dataset_outputs]) / len(self.bert_single_dataset_outputs)

        return loss


@dataclass
class Data_Collator():

    @staticmethod
    def collate_batch(features: List[InputFeatures]) -> BertBatchInput:
        """Overrides parent method. Turns features of a Batch into a BertBatchInput object,
         thereby creating tensors with the correct type.

        It takes the first element of each data set in a batch as a proxy for what attributes exist on the whole batch.
        Different to the parent method because it evaluates on tasks and data sets as well.
        :param features: List of features that are applied to the entire batch.
        :return batch: BertBatchInput of the batch features.
        """

        # we split for each data set
        split_features_dict = defaultdict(list)
        for feature in features:
            split_features_dict[feature.data_set].append(feature)
        bert_inputs = []

        for data_set, feature_split in split_features_dict.items():
            # within each dataset we _do_ assume common attributes
            first = feature_split[0]
            # LABEL: this should be done automatically, but we make sure the correct label type is assigned
            if hasattr(first, "label") and first.label is not None:
                if type(first.label) is int:
                    labels = torch.tensor([f.label for f in feature_split], dtype=torch.long)
                else:
                    labels = torch.tensor([f.label for f in feature_split], dtype=torch.float)
            else:
                labels = None

            # Other (mandatory) attributes
            attention_mask = torch.tensor([f.attention_mask for f in feature_split], dtype=torch.long)
            input_ids = torch.tensor([f.input_ids for f in feature_split], dtype=torch.long)

            data_set = first.data_set
            bert_inputs.append(BertSingleDatasetInput(labels, attention_mask, input_ids, data_set))

        return BertBatchInput(bert_inputs)


@dataclass()
class TestResult:
    data_set: str
    predictions: List
    true_labels: List


class RegressionResult:
    """Saves the regression results of the dataset classification"""

    def __init__(self, dataset_name, pearson_score, spearman_score, corr, dataset_size):
        """A set of regression results for a dataset evaluation run.

        :param dataset_name: Name of the dataset for which these results are for.
        :param pearson_score: Pearson correlation score
        :param spearman_score: Spearman correlation coefficient
        :param corr: Correlation
        :param dataset_size: Size of the dataset
        :param task: Task for which the dataset was evaluated
        """
        self.dataset_name = dataset_name
        self.pearson_score = pearson_score
        self.spearman_score = spearman_score
        self.corr = corr
        self.dataset_size = dataset_size


def split_features_by_dataset(features) -> defaultdict:
    """ Split features of each data set.

    :param features: List of the InputFeatures.
    :return d: dictionary containing a mapping of each feature of a dataset to feature.
    """
    d = defaultdict(list)
    for feature in features:
        d[feature.data_set].append(feature)
    return d


class Split(Enum):
    """Possible dataset split values.

    'train' are intended for model training
    'dev' are intended for model validation
    'test' are intended for evaluating the model
    """
    train = "train"
    dev = "dev"
    test = "test"


class Columns(Enum):
    """Mapping of info to column of the preprocessed data set."""
    ID = 0
    SENTENCE = 1
    LABEL = 2
    SET = 3


class DataFile:

    def __init__(self, data_path: str):
        """ Instantiates a DataFile object to store attributes of a file.

        :param data_path: Path to where the data file is saved.
        :param scenario: Scenario split that is of interest.
        """
        self.data_path = data_path
        self.data_lines, self.dataset_length = self._read_csv()
        self.dataset_name = os.path.basename(self.data_path)

    def _read_csv(self, quotechar: str = '"') -> List:
        """Reads a comma separated value file. Returns list of read csv lines

        :param quotechar: The character the csv will use to separate sentences.
        :return lines: A list of each row in the csv that is represented as another list.
        :return data_length: data lines and dictionary of the number of the dev, test, and train data lines.
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            lines = list(csv.reader(f, delimiter=",", quotechar=quotechar))
            self.df = pd.read_csv(f.name)
            dataset_length = self.df.groupby(self.df["set"])['id'].count().to_dict()

            return lines, dataset_length


def load_data_files(dataset_folder: str) -> List[DataFile]:

    dataset_paths = "/mnt/data2/Sid/arg_quality/pytorch/task4/data/*.csv"
    datasets = glob.glob(dataset_paths)
    datasets_list = [DataFile(dataset) for dataset in datasets]

    return datasets_list



