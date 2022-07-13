# Standard Imports
import os
import glob


def generate_task_list(hparams) -> list:
    """

    :param hparams:
    :return:
    """
    datasets = glob.glob(hparams.data_folder_path)
    dataset_names = [os.path.basename(data_path)[:-4]
                     for data_path in datasets]
    # for single dataset "only" keyword is used
    if "only" in hparams.task_name:
        task_dict = [dataset_name for dataset_name in dataset_names
                     if dataset_name in hparams.task_name]
    elif "LOO_" in hparams.task_name:
        task_dict = [dataset_name for dataset_name in dataset_names
                     if dataset_name not in hparams.task_name]
    else:
        task_dict = dataset_names

    return task_dict
