from __future__ import absolute_import, division, print_function

import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (BertConfig, BertModel, BertPreTrainedModel, get_linear_schedule_with_warmup,
                          BertTokenizer, AdamW
                          )
from tqdm import tqdm
from scipy.stats import spearmanr

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import csv
import random
import argparse
import mlflow


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class Trainer(BertPreTrainedModel):
    def __init__(self, config):

        super(Trainer, self).__init__(config)

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3072, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        """

        :param input_ids: Indices of input sequence tokens in the vocabulary
        :param token_type_ids: Segment token indices to indicate first and second portions of the inputs
        :param attention_mask: Mask to avoid performing attention on padding token indices

        :returns:
            outputs: List containing loss, prediction scores, hidden states and attentions

        """
        outputs = self.bert(input_ids,
                            token_type_ids,
                            attention_mask)

        hidden_states = outputs[-1]
        out = torch.cat(tuple([hidden_states[i] for i in [-1, -2, -3, -4]]), dim=-1)

        # Pooling by also setting masked items to zero
        bert_mask = attention_mask.unsqueeze(2)
        # Multiply output with mask to only retain non-padding tokens
        out = torch.mul(out, bert_mask)

        # First item ['CLS'] is sentence representation
        out = out[:, 0, :]

        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(out.view(-1), labels.view(-1))

            outputs = [loss] + [out.tolist()]
            return outputs
        else:
            return out


class DataPreprocessForSingleSentence(object):

    def __init__(self, bert_tokenizer, max_workers=10):
        self.bert_tokenizer = bert_tokenizer
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def get_input(self, dataset, max_seq_len):
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].astype('float64', copy=True).tolist()
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels

    def trunate_and_pad(self, seq, max_seq_len):
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        seq = ['[CLS]'] + seq + ['[SEP]']
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = [0] * len(seq) + padding
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def gen_dataloader(tokenizer, data_file, max_seq_len, batch_size):
    processor = DataPreprocessForSingleSentence(bert_tokenizer=tokenizer)
    data = pd.read_csv(data_file, sep='\t')
    seqs, seq_masks, seq_segments, labels = processor.get_input(
        dataset=data, max_seq_len=max_seq_len)
    seqs = torch.tensor(seqs, dtype=torch.long)
    seq_masks = torch.tensor(seq_masks, dtype=torch.long)
    seq_segments = torch.tensor(seq_segments, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    data = TensorDataset(seqs, seq_masks, seq_segments, labels)
    dataloader = DataLoader(dataset=data, batch_size=batch_size)
    return dataloader


def test(arg, device):

    # load model
    tokenizer = BertTokenizer.from_pretrained(args.outdir_model, do_lower_case=False)
    model = Trainer.from_pretrained(arg.outdir_model)
    model.to(device)
    
    # test the model on all test sets
    filenames = glob.glob("test/*.txt")
    for filename in filenames:
        test_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=filename,
                                         max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)

        true_labels = []
        pred_labels = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_segments, b_labels = batch
                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs
                # print(len(logits))
                pred_labels.append(logits.detach().cpu().numpy())
                true_labels.append(b_labels.to('cpu').numpy())
        prediction = np.concatenate(pred_labels).flat
        truth = np.concatenate(true_labels, axis=0)

        pearson_corr = np.corrcoef(truth, prediction)
        spearman_corr, p = spearmanr(truth, prediction)

        mlflow.log_metric("pearson corr", pearson_corr[0][1])
        mlflow.log_metric("spearman corr", spearman_corr)
        list1 = [filename, arg.seed, pearson_corr[0][1], spearman_corr]

        # print(pearson_corr[0][1])
        path_eval = os.path.join(arg.setup, "eval_results.csv")
        df = pd.DataFrame([list1])
        if not os.path.exists(path_eval):
            df.to_csv(path_eval, sep=",", header=["test_file", "seed", "pearson", "spearman"], index=False)
        else:
            df.to_csv(path_eval, index=False, mode='a', header=False)


def train(arg, device):

    VOCAB = 'vocab.txt'
    if arg.finetuned:
        tokenizer = Trainer.from_pretrained(os.path.join(arg.model, VOCAB), do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(arg.model, do_lower_case=False)
    config = BertConfig.from_pretrained(arg.model)
    model = Trainer.from_pretrained(arg.model,
                                        from_tf=False,
                                        config=config)
    model.to(device)
    lr = arg.learning_rate
    num_total_steps = 1000
    num_warmup_steps = 100

    train_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.traindata_file,
                                      max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)
    validation_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.devdata_file,
                                           max_seq_len=arg.max_seq_length, batch_size=arg.batch_size)

    ### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,  # Default value in run_glue.py
                                                num_training_steps=num_total_steps)

    epochs = arg.epochs
    patience = arg.patience
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # Store our loss and accuracy for plotting
    valid_losses = []
    train_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    # trange is a tqdm wrapper around the normal python range
    for epoch in tqdm(range(epochs)):

        # Training, Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # total_step = len(train_dataloader)
        # Train the data for one epoch
        for i, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_seq_segments, b_labels = batch
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            train_losses.append(loss.item())
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        # test model on validation set
        model.eval()
        with torch.no_grad():

            for i, batch in enumerate(validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_segments, b_labels = batch
                # Forward pass
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                valid_losses.append(loss.item())


        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        mlflow.log_metric("train_loss", train_loss, epoch)
        mlflow.log_metric("validation_loss", valid_loss, epoch)

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
        valid_losses = []
        train_losses = []

    # save training and validation losses
    tr_losses_path = "training_losses" + str(arg.seed) + ".tsv"
    val_losses_path = "validation_losses" + str(arg.seed) + ".tsv"
    np.savetxt(tr_losses_path, avg_train_losses, delimiter='\t')
    np.savetxt(val_losses_path, avg_valid_losses,  delimiter='\t')

    # save model
    if arg.save_model:
        model.save_pretrained(arg.outdir_model)
        tokenizer.save_pretrained(arg.outdir_model)
        print("saving model")


def predict(arg, device):
    # VOCAB = 'vocab.txt'
    # if arg.finetuned:
    #     tokenizer = BertTokenizer.from_pretrained(os.path.join(arg.model, VOCAB), do_lower_case=False)
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(arg.model, do_lower_case=False)
    #
    # test_dataloader = gen_dataloader(tokenizer=tokenizer, data_file=arg.testdata_file, max_seq_len=arg.max_seq_length,
    #                                  batch_size=arg.batch_size)
    # # :TODO: complete model
    # model = Trainer()
    # model.to(device)
    #
    # # test model on test set
    # path_pred = arg.prediction_file
    # f = open(path_pred, 'w+', newline='')
    # writer = csv.writer(f)
    # writer.writerow(['label'])
    #
    # pred_labels = []
    # model.eval()
    # with torch.no_grad():
    #     for i, batch in enumerate(test_dataloader):
    #         batch = tuple(t.to(device) for t in batch)
    #         # Unpack the inputs from our dataloader
    #         b_input_ids, b_input_mask, b_segments, b_labels = batch
    #         # Forward pass
    #         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    #         logits = outputs[0]
    #         pred_labels.append(logits.cpu().numpy())
    # prediction = np.concatenate(pred_labels, axis=0)
    # for i in range(0, len(prediction)):
    #     writer.writerow([prediction[i][0]])
    return None

def set_seed(seed=3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--setup', type=str)
    parser.add_argument('--max_seq_length', default=85, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--traindata_file', default=None, type=str)
    parser.add_argument('--devdata_file', default=None, type=str)
    parser.add_argument('--testdata_folder', default=None, type=str)
    parser.add_argument('--model', default="bert-base-uncased", type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--seed', default=42, type=float)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--outdir_model', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--finetuned', default=False, type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--prediction_file', type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print("Using Device:", device)
    print("GPUs:", n_gpu)

    mlflow.set_tracking_uri("http://mlflow.dbs.ifi.lmu.de:5000")
    mlflow.set_experiment(experiment_name="Sid_task3_exp1")
    with mlflow.start_run():
        for seed in [1212]:
            args.seed = seed
            set_seed(args.seed)

            for setup in ["all"]:
                args.setup = setup
                train_filename = "data/" + setup + "_train.txt"
                dev_filename = "data/" + setup + "_dev.txt"

                args.traindata_file = os.path.join(setup, train_filename)
                args.devdata_file = os.path.join(setup, dev_filename)
                args.testdata_folder = "test"
                args.outdir_model = os.path.join(setup, "model" + str(seed))
                if not os.path.exists(args.outdir_model):
                    os.makedirs(args.outdir_model)

                if args.task == 'train':
                    train(args, device)
                    test(args, device)

                elif args.task == 'test':
                    test(args, device)

                else:
                    predict(args, device)
