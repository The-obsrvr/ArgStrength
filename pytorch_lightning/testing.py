"""
Test the trained model
"""
import os
from argparse import ArgumentParser, Namespace

import pandas as pd
import yaml
from tqdm import tqdm

from .regressor import ArgStrRanker
from .datasets import ArgumentDataset


def load_model_from_experiment(experiment_folder: str):
    """

    :param experiment_folder:
    :return:
    """

    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    model = ArgStrRanker.load_from_checkpoint(
        checkpoint_path, hparams=Namespace(**hparams)
    )

    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Path to the experiment folder.",
    )
    parser.add_argument(
        "--test_data",
        required=True,
        type=str,
        help="Path to the test data.",
    )
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)

    test_set = ArgumentDataset(model.hparams,
                               mode="test",
                               tokenizer=model.tokenizer,
                               )
    predictions = [
        model.predict(sample)
        for sample in tqdm(test_set, desc="Testing on {}".format(hparams.test_data))
    ]

    y_pred = [o["predicted_label"] for o in predictions]
    y_true = [s["label"] for s in test_set]
    datasets = [s["dataset"] for s in test_set]

    # define the performance metrics here. Store in a csv file.
