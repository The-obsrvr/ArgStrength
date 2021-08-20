"""Model, Data and Training Arguments DataClasses."""

# Standard Imports
from datetime import datetime
from os import path
# Third Party Imports
from dataclasses import dataclass, field


# noinspection PyCompatibility
@dataclass
class ModelDataArguments:
    """ Arguments pertaining to running the model.

    Arguments pertaining to how our model will be saved, what data to input
    into the model, and scenarios/tasks to train and evaluate for.
    MLFlow tracking settings and training task settings.
    Using `HfArgumentParser` we can turn this class into argparse arguments to
    be able to specify them on the command line.
    """
    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "The (pre-trained) model to use. (examples: bert-base-uncased, bert-base-cased)"
                          " Default: 'bert-base-uncased'"}
    )
    task_name: str = field(
        default="MTLAS",
        metadata={"help": "The experimental set up adopted by the current experiment. (examples: MTLAS, STLAS, etc)"
                  "Default: 'MTLAS"}
    )
    data_dir: str = field(
        default=path.join(path.dirname(path.dirname(path.abspath(__file__))), 'preprocessed'),
        metadata={"help": "The input data dir. Should contain the .csv files (or other data files) for the task."
                          " Default: 'src/preprocessed/'"}
    )
    max_seq_length_perc: float = field(
        default=0.95,
        metadata={"help": "Determines the sequence length as percentile of sentence lengths. "
                          "Sequences longer than calculated max_seq_length will be truncated, "
                          "sequences shorter will be padded. Default = 0.95"}
    )
    mlflow_run: bool = field(
        default=True,
        metadata={"help": "If MlFlow tracking should be used. "
                          "Requires a connection to the MlFlow Server. "
                          "Use no-mlflow_run to turn off. "
                          "Default: 'True'"}
    )
    mlflow_exp_name: str = field(
        default="AM-DU " + str(datetime.now()),
        metadata={"help": "The experiment name under which the MlFlow params, metrics, artifacts should be tracked."
                          " Default: 'AM-DU <current-date-time>'"}
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets."
                                        " Default: 'True', to turn off, use '--no-overwrite_cache'"}
    )
    log_artifacts_to_remote: bool = field(
        default=False, metadata={"help": "If artifacts (model files) should be saved onto the remote MLflow server."
                                         " Default: 'False' Use --log_artifacts_to_remote to enable."}
    )
    log_artifacts_to_local: bool = field(
        default=False, metadata={"help": "If artifacts (model files) should be saved onto the local MLflow server."
                                         "Default: 'False' Use --log_artifacts_to_local to enable."}
    )


# noinspection PyCompatibility
@dataclass
class TrainingArguments:
    """Arguments pertaining to the training of the model.

    TrainingArguments is the subset of the arguments we use in our example
    scripts **which relate to the training loop itself**.
    Using `HfArgumentParser` we can turn this class into argparse arguments to
    be able to specify them on the command line.
    """

    output_dir: str = field(
        default=path.join(path.dirname(path.dirname(path.abspath(__file__))), 'model'),
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."
                          " Default: /src/model/"}
    )
    load_model: str = field(
        default=None,
        metadata={"help": "The absolute filepath to pre-trained model to load. Default: None"}
    )
    train_model: bool = field(
        default=True,
        metadata={"help": "Set to false if the dataset should only be used to evaluate the model. Default: True"}
    )
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "Set to True if eval is to be run after every specified intervals during training. "
                          "Default: True"}
    )
    eval_steps: int = field(
        default=1,
        metadata={"help": "Set to step number after which evaluation is to be run"}
    )
    save_steps: int = field(
        default=0,
        metadata={"help": "Set step number at which checkpoint has to be saved."}
    )
    weighted_dataset_loss: str = field(
        default="equal", metadata={"help": "Choose between three possible weighted loss based on data set: unweighted,"
                                           " equal, weighted."}
    )
    train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training. Default: 64"}
    )
    eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation. Default: 128"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for Adam. Default: '2e-5'"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer. Default: '1e-8'"})
    weight_decay: float = field(default=0.1, metadata={"help": "Learning Rate weight decay."})
    max_num_train_epochs: float = field(default=20.0, metadata={"help": "Maximum number of training epochs to perform. "
                                                                        "Early Stopping should kick in before."
                                                                        "Default: 20"})
    patience: int = field(default=5, metadata={"help": "Patience for early stopping. Default: 5.0"})
    logging_steps: int = field(default=-2, metadata={"help": "Log every X updates steps."})
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    is_distributed: bool = field(default=False, metadata={"help": "whether distributed training is enabled. "
                                                                  "Default: False"})
    gpus_per_trial: int = field(default=2, metadata={"help": "How many GPUs to use for training. Default: 2"})
    gpu_device: int = field(default=3, metadata={"help": "Which gpu to use for training. Default: 0"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization. Default: 42"})
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank. Default: -1"})
