# Local Imports
from utils import BertBatchInput, BertSingleDatasetOutput, BertBatchOutput
# Standard Imports

# Third Party Imports
import torch
from torch import nn
from torch.nn import MSELoss
from transformers import BertPreTrainedModel, BertModel


class BertForSequenceClassification(BertPreTrainedModel):
    """Handles weights and classifier initialization. Adjusted forward pass to allow for multi task learning."""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.regressors = {}

    def get_or_create_regressor(self, dataset, device):
        """Creates or returns already created classifier for a task.

        :param task:
            Task to train on ex. ArgNonarg, ProConNon, etc.
        :param device:
            Device (CPU or GPU) that will be used for training
        :return: Classifier that will be used for the forward pass.
        """
        if self.regressors.get(dataset) is not None:
            return self.regressors[dataset]
        else:
            self.regressors[dataset] = nn.Sequential(
                nn.Linear(3072, 100),
                nn.ReLU(),
                nn.Linear(100, 1),
                nn.Sigmoid()
            ).to(device)
            return self.regressors[dataset]

    def forward(
        self,
        bert_batch_input: BertBatchInput,
        calculate_loss: bool = True
    ):
        """
        Performs a forward pass. In particular, separates output logits for each dataset and uses a task-specific
        classification head (multi-task learning). Outputs are separated and returned for each dataset.

        :param bert_batch_input:
            The input infos needed to perform a forward pass (task, features, ...)
        :param calculate_loss:
            If the Cross Entropy calculation should be label-weighted.
        """
        single_dataset_outputs = []

        for bert_batch_input in bert_batch_input.bert_single_dataset_inputs:
            labels = bert_batch_input.labels
            data_set = bert_batch_input.data_set
            outputs = self.bert(
                bert_batch_input.input_ids,
                attention_mask=bert_batch_input.attention_mask,
            )

            hidden_states = outputs[-1]
            out = torch.cat(tuple([hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)

            # Pooling by also setting masked items to zero
            bert_mask = bert_batch_input.attention_mask.unsqueeze(2)
            # Multiply output with mask to only retain non-padding tokens
            out = torch.mul(out, bert_mask)

            # First item ['CLS'] is sentence representation
            out = out[:, 0, :]

            # Get the task-specific classifier
            regressor = self.get_or_create_regressor(data_set, out.device)
            # Get the logits
            logits = regressor(out)

            if calculate_loss:
                # We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = None
            single_dataset_outputs.append(BertSingleDatasetOutput(labels=labels, logits=logits,
                                                                  loss=loss, data_set=data_set))

        return BertBatchOutput(single_dataset_outputs)
