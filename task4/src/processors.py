"""
contains function that read input data files and generates features ~ batches

"""
# local Imports
from utils import InputExample, InputFeatures, DataFile, Split, Columns

from typing import List
from abc import abstractmethod


class StrengthProcessor:
    def __init__(self, data_files: List[DataFile]):
        """
        Abstract processor with initialization for preprocessed data_files. Provides methods to create
        examples for training/validation/testing. Processors differ by their task (labels, output_mode,...).

        :param data_files:
            data files for this processor
        """
        self.data_files = data_files

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "train")

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "dev")

    def get_test_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(self.data_files, "test")

    def _create_examples(self, data_files: List[DataFile], set_type: str) -> List[InputExample]:
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
            for (i, line) in enumerate(data_file.data_lines):
                line_type = line[Columns.SET.value]
                if i == 0 or not line_type == set_type:
                    continue
                guid = "%s-%s" % (set_type, new_id)
                new_id += 1
                text_a = line[Columns.SENTENCE.value]
                # topic can be added here as text_b
                text_b = None
                label = line[Columns.LABEL.value]
                data_set = data_file.dataset_name
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                             data_set=data_set))
        return examples

    @staticmethod
    @abstractmethod
    def get_task_name():
        """Returns the name of the task the processor handles"""
        return "ArgumentStrength"

    @staticmethod
    @abstractmethod
    def get_output_mode():
        """Returns output mode of the task (classification/regression)"""
        return "regression"


# Combine lines of all data sets
def get_combined_lines(StrengthProcessor) -> List:
    """Method to combine all dataset lines of a processor"""
    combined_lines = []
    for data_file in StrengthProcessor.data_files:
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
    :param label_map:
        The label mapping (i.e. from label names to integers for classification)
    :param output_mode:
        The output_mode of examples (i.e. Regression or Classifiation)

    :returns:
        Tokenized Input Features from passed data examples.
    """
    if max_length is None:
        max_length = tokenizer.max_len

    labels = [float(example.label) if example.label is not None else None for example in examples]
    data_sets = [example.data_set for example in examples]

    # The actual encoding of sentences, with padding and max length! Text_a = sentences, text_b = Topics!
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i], data_set=data_sets[i])
        features.append(feature)

    return features


def convert_features_to_dataset(StrengthProcessor, tokenizer, max_token_length, mode):
    """
    Create the datasets
    :return: train/dev/test dataset
    """

    if mode == Split.dev:
        features = _convert_examples_to_features(StrengthProcessor.get_dev_examples(), tokenizer,
                                                 max_token_length)
    elif mode == Split.test:
        features = _convert_examples_to_features(StrengthProcessor.get_test_examples(), tokenizer,
                                                 max_token_length)
    else:
        features = _convert_examples_to_features(StrengthProcessor.get_train_examples(), tokenizer,
                                                 max_token_length)

    return features
