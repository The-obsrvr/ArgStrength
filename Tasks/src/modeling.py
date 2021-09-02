# Local Imports
from utils import BertBatchInput, BertSingleDatasetOutput, BertBatchOutput
# Standard Imports

# Third Party Imports
import torch
from torch import nn
from torch.nn import MSELoss
from transformers import BertPreTrainedModel, BertModel


class ArgStrModel(BertPreTrainedModel):
    """Handles weights and regressor initialization. Adjusted forward pass to allow for multi task learning."""

    def __init__(self, config=None, dropout_prob=0.2, bert_hidden_layers=None,
                 mlp_config=None, task_dict=None, device=None):
        super(ArgStrModel, self).__init__(config)

        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.dropout_prob = dropout_prob
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bert_hidden_layers = bert_hidden_layers
        self.mlp_config = mlp_config
        self.task_name = config.finetuning_task
        if "MTLAS" in config.finetuning_task:
            self.regressors = nn.ModuleDict()
            for dataset in task_dict:
                self.regressors[dataset] = self.generate_regressors(device)
        else:
            self.regressor = self.generate_regressors(device)

    def generate_regressors(self, device):
        """

        :param device:
        :return:
        """

        last_dim = self.hidden_size * self.bert_hidden_layers
        layers = []
        if self.mlp_config == 1:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            last_dim = 512
        elif self.mlp_config == 2:
            layers.append(nn.Linear(in_features=last_dim, out_features=100, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=100))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            last_dim = 100
        elif self.mlp_config == 3:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            layers.append(nn.Linear(in_features=512, out_features=100, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=100))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            last_dim = 100
        else:
            layers.append(nn.Linear(in_features=last_dim, out_features=512, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=512))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            layers.append(nn.Linear(in_features=512, out_features=256, bias=False))
            # layers.append(nn.BatchNorm1d(num_features=256))
            layers.append(nn.ReLU())
            if self.dropout_prob is not None:
                layers.append(nn.Dropout(p=self.dropout_prob))
            last_dim = 256
        layers.append(nn.Linear(in_features=last_dim, out_features=1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers).to(device)

    def get_or_create_regressor(self, dataset, device):
        """Creates or returns already created regressor for a task.

        :param dataset:
            Dataset being processed: gretz, toledo, swanson, UKPRank.
        :param device:
            Device (CPU or GPU) that will be used for training
        :return: Regressor that will be used for the forward pass.
        """
        if self.regressors[dataset] is not None:
            return self.regressors[dataset]
        else:
            self.regressors[dataset] = self.generate_regressors(device)
            return self.regressors[dataset]

    def forward(
            self,
            bert_batch_input: BertBatchInput,
            calculate_loss: bool = True
    ):
        """
        Performs a forward pass. In particular, separates output logits for each dataset and uses a dataset-specific
        regression head (multi-task learning). Outputs are separated and returned for each dataset.

        :param bert_batch_input:
            The input infos needed to perform a forward pass (dataset, features, ...)
        :param calculate_loss:
            To calculate the MSELoss between the logits and losses.
        :returns:
            BertBatchOutput
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
            out = torch.cat(tuple([hidden_states[-i] for i in range(1, self.bert_hidden_layers + 1)]), dim=-1)

            # Pooling by also setting masked items to zero
            bert_mask = bert_batch_input.attention_mask.unsqueeze(2)
            # Multiply output with mask to only retain non-padding tokens
            out = torch.mul(out, bert_mask)

            # First item ['CLS'] is sentence representation
            out = out[:, 0, :]

            # Get the task-specific classifier
            if "MTLAS" in self.task_name:
                regressor = self.get_or_create_regressor(data_set, out.device)
            else:
                regressor = self.regressor
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

    def forward_for_inference(
            self,
            bert_batch_input: BertBatchInput,
    ):
        """

        :param bert_batch_input:
            BertBatchInput containing input_ids, attention_mask and labels info.
        :return:
           modified BertBatchOutput
        """

        per_batch_output = []
        for bert_batch_input in bert_batch_input.bert_single_dataset_inputs:
            labels = bert_batch_input.labels
            dataset = bert_batch_input.data_set
            outputs = self.bert(
                bert_batch_input.input_ids,
                attention_mask=bert_batch_input.attention_mask
            )

            hidden_states = outputs[-1]
            out = torch.cat(tuple([hidden_states[-i] for i in range(1, self.bert_hidden_layers + 1)]), dim=-1)

            # Pooling by also setting masked items to zero
            bert_mask = bert_batch_input.attention_mask.unsqueeze(2)
            # Multiply output with mask to only retain non-padding tokens
            out = torch.mul(out, bert_mask)

            # Extract First item ['CLS'] i.e. the sentence representation
            out = out[:, 0, :]

            logits = []
            # run the out through each of the multi-regressor heads
            for regressor in self.regressors:
                regressor_unit = self.get_or_create_regressor(regressor, out.device)
                # gets the logit value for this regressor and store in the list
                logits.append(regressor_unit(out).to('cpu').numpy())

            single_output = [labels, logits, dataset]
            per_batch_output.append(single_output)

        return per_batch_output
