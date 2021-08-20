from __future__ import absolute_import, division, print_function

# Local Imports
from modeling import ArgStrModel
from arguments import TrainingArguments
from training import ArgStrTrainer
from processors import get_datasets
from utils import pearson_and_spearman, Data_Collator

# Standard Imports
import os

# Third Party Imports
import json
import torch

from transformers import BertConfig

if __name__ == '__main__':
    # load the best trained model and config

    config_file = open("/mnt/data2/Sid/arg_quality/pytorch/task4_hpo/best_model_details_MTLAS_LOO_swanson_v2_bb.json", )
    config_data = json.load(config_file)

    train_dataset, eval_dataset, _, task_dict = get_datasets(config_data)
    print(task_dict)
    bert_config = BertConfig.from_pretrained(
        config_data["bert_arch"],
        finetuning_task=config_data["task_name"],
        output_hidden_states=True
    )
    optimised_bert_model = ArgStrModel(
        config=bert_config,
        dropout_prob=config_data["dropout_prob"],
        bert_hidden_layers=config_data["bert_hidden_layers"],
        mlp_config=config_data["nn_config"],
        task_dict=task_dict,
        device=torch.device('cuda'
                            if torch.cuda.is_available()
                            else 'cpu')
    )
    with open(os.path.join(config_data["best_checkpoint_path"], "best_model.pt"), 'rb') as checkpt:
        model_state, optimizer_state = torch.load(checkpt)
    optimised_bert_model.load_state_dict(model_state)
    # print(optimised_bert_model)

    training_args = TrainingArguments(
        learning_rate=config_data["learning_rate"],
        train_model=True,
        evaluate_during_training=True,
        save_steps=0,
        max_num_train_epochs=8,
        train_batch_size=config_data["train_batch_size"],
        eval_batch_size=config_data["eval_batch_size"],
        weight_decay=config_data["weight_decay"],
        weighted_dataset_loss=config_data["dataset_loss_method"]
    )

    train_runner = ArgStrTrainer(
        optimised_bert_model,
        training_args,
        train_dataset=train_dataset,
        data_collator=Data_Collator,
        task_name=config_data["task_name"]
    )

    train_runner.train_model(optimizer_state=optimizer_state, device=config_data["device"], mlflow_logging=False)
