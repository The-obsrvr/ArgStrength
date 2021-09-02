"""
contains function that read input data files and generates features ~ batches

"""
# local Imports
from utils import InputExample, InputFeatures, DataFile, Split, Columns, load_data_files

# Standard Imports
from typing import List
from abc import abstractmethod

# Third Party Imports
import numpy as np
import mlflow
from transformers import BertTokenizer


class StrengthProcessor:
    def __init__(self, data_files: List[DataFile], task_name: str):
        """
        Abstract processor with initialization for preprocessed data_files. Provides methods to create
        examples for training/validation/testing. Processors differ by their task (labels, output_mode,...).

        :param data_files:
            data files for this processor
        :param task_name:
            name of the task to be performed
        """
        self.task_name = task_name
        self.data_files = data_files

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "train", self.task_name)

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "dev", self.task_name)

    def get_test_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "test", task_name=self.task_name)

    @staticmethod
    def _create_examples(data_files: List[DataFile],
                         set_type: str = "train",
                         task_name: str = None) -> List[InputExample]:
        """
        Creates examples for the training, dev and test sets.

        :param data_files:
            data files to extract the rows from
        :param set_type:
            the type of rows to extract (train/dev/test)

        :returns:
            examples for all matching rows in the data files.
        """
        examples = []
        new_id = 0
        for data_file in data_files:

            if str("STLAS_LOO_" + data_file.dataset_name) in task_name and set_type != "test":
                continue
            if str("MTLAS_LOO_" + data_file.dataset_name) in task_name:
                continue
            if "only" in task_name and str("_" + data_file.dataset_name) not in task_name and set_type != "test":
                continue

            for (i, line) in enumerate(data_file.data_lines):
                line_type = line[Columns.SET.value]
                if i == 0 or not line_type == set_type:
                    continue

                guid = "%s-%s" % (set_type, new_id)
                new_id += 1
                text_a = line[Columns.SENTENCE.value]
                # topic can be added here as text_b
                if "source" in task_name:
                    text_b = data_file.dataset_name
                elif "topic" in task_name:
                    text_b = line[Columns.TOPIC.value]
                else:
                    text_b = None
                label = line[Columns.LABEL.value]
                data_set = data_file.dataset_name
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                             data_set=data_set))
        return examples

    def get_task_name(self):
        """Returns the name of the task the processor handles"""
        return self.task_name


# Combine lines of all data sets
def get_combined_lines(processor) -> List:
    """
    Method to combine all dataset lines of a processor
    :returns: list of all the lines from all the data sets
    """
    combined_lines = []
    for data_file in processor.data_files:
        combined_lines += (data_file.data_lines[1:])
    return combined_lines


def _convert_examples_to_features(
        examples: List[InputExample],
        tokenizer,
        max_length: int,
) -> List[InputFeatures]:
    """
    Converts dataset examples to InputFeatures

    :param examples:
        List of InputExamples to transform.
    :param tokenizer:
        The tokenizer to tokenize sentences + topics.
    :param max_length:
        The max token length.
    :returns:
        Tokenized Input Features from passed data examples.
    """
    if max_length is None:
        max_length = tokenizer.max_len

    labels = [float(example.label) if example.label is not None else None for example in examples]
    data_sets = [example.data_set for example in examples]

    # The actual encoding of sentences, with padding and max length! Text_a = sentences, text_b = Topics~ dataset_name!
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i], data_set=data_sets[i])
        features.append(feature)

    return features


def convert_features_to_dataset(processor, tokenizer, mode, max_token_length=None, max_token_length_perc=0.95):
    """
    Create the datasets
    :return: train/dev/test dataset
    """
    if max_token_length is None:
        max_token_length = find_max_length(tokenizer, processor, p=max_token_length_perc)
        mlflow.log_param("max token length", max_token_length)

    if mode == "dev":
        features = _convert_examples_to_features(processor.get_dev_examples(), tokenizer,
                                                 max_token_length)
    elif mode == "test":
        features = _convert_examples_to_features(processor.get_test_examples(), tokenizer,
                                                 max_token_length)
    else:
        features = _convert_examples_to_features(processor.get_train_examples(), tokenizer,
                                                 max_token_length)

    return features


def find_max_length(tokenizer, processor, p: float) -> int:
    """
    :tokenizer: BERT tokenizer used for tokenizing the data
    :processor: The Processor containing the data files
    :p: percentile used to decide the maximum token length to be used.

    :return: maximum token length to be used
    """
    combined_lines = get_combined_lines(processor)
    sens = [(line[Columns.SENTENCE.value]) for line in combined_lines]
    batch_encoding = tokenizer.batch_encode_plus(
        sens, pad_to_max_length=False,
    )
    # Collect all lengths
    lens = [len(x) for x in batch_encoding['input_ids']]
    # Get the p percentile of lengths.
    res = int(np.percentile(lens, p * 100))

    return res


def get_datasets(config):
    data_files, task_list = load_data_files(config["data_dir"],
                                            task_name=config["task_name"],
                                            split_by_topic=config["split_by_topic"])
    # initialize the Processor
    processor = StrengthProcessor(data_files, task_name=config["task_name"])
    if "cased" in config["bert_arch"]:
        tokenizer = BertTokenizer.from_pretrained(config["bert_arch"], do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(config["bert_arch"], do_lower_case=False)

    # prepare the train and validation datasets
    train_dataset = convert_features_to_dataset(processor, tokenizer,
                                                "train", config["max_seq_length"], config["max_seq_length_perc"])
    validation_dataset = convert_features_to_dataset(processor, tokenizer,
                                                     "dev", config["max_seq_length"], config["max_seq_length_perc"])
    test_dataset = convert_features_to_dataset(processor, tokenizer,
                                               "test", config["max_seq_length"], config["max_seq_length_perc"])

    mlflow.log_param("length_of_training", len(train_dataset))
    mlflow.log_param("length_of_validation", len(validation_dataset))
    mlflow.log_param("length_of_test", len(test_dataset))

    return train_dataset, validation_dataset, test_dataset, task_list
