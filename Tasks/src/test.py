import torch
from modeling import MultiTaskArgRanker
from transformers import BertConfig

bert_config = BertConfig.from_pretrained(
    "bert-base-uncased",
    finetuning_task="asb",
    output_hidden_states=True
)

model = MultiTaskArgRanker.from_pretrained(
    "bert-base-uncased",
    config=bert_config,
    dropout_prob=0.2,
    bert_hidden_layers=4,
    mlp_config=2,
    device=torch.device('cuda')
)

print(model)