# coding=utf8

"""
this file is designed to run wsc bench mark
"""

import os
import argparse
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import dataloader
from torch import nn
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tokenizer import PretrainedTokenizer
from wsc_dataset import SpanWscDataset
from wsc_data_prep import WscDataPreper
from wsc_model import WscModel


torch.multiprocessing.set_sharing_strategy('file_system')


class WscTrainer(object):
    def __init__(self, config):
        self.tokenizer = PretrainedTokenizer(config.pretrain_name)
        self.data_prep = WscDataPreper()
        cls_num = self.data_prep.get_cls_num()
        config.cls_num = cls_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.model = WscModel(config, device=self.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.writer = SummaryWriter(log_path)
        # self.metric_calcutor = self.create_metric_calculator()

    def train(self):
        model_dir = os.path.dirname(self.config.model_path)
        os.makedirs(model_dir, exist_ok=True)
        train_dl = self.get_dataloader(self.config.train_path, self.config.batch_size)
        val_dl = self.get_dataloader(self.config.valid_path, self.config.batch_size)
        optimizer = self.create_optimizer(self.config.weight_decay, self.config.lr)
        epochs = self.config.epochs
        total_steps = epochs * len(train_dl)
        lr_scheduler = self.create_scheduler(optimizer,  total_steps, self.config.warm_up_ratio, None)
        best_acc = 0.
        for epoch in tqdm(range(epochs)):
            self.train_epoch(train_dl, optimizer, lr_scheduler, self.criterion, epoch)
            acc = self.evalute(val_dl)
            self.print_metric(False, epoch, acc, best_acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model, self.config.model_path)
            acc = self.evalute(train_dl)
            self.print_metric(True, epoch, acc, best_acc)

    def train_epoch(self, dl, optimizer, lr_scheduler, criterion, epoch):
        self.model.train()
        for step, batch in tqdm(enumerate(dl), desc=f"itering in epoch: {epoch}"):
            self.model.zero_grad()
            x = batch["input_ids"]
            y = batch["tags"]
            y = y.to(self.device)
            logits = self.model(x, **batch)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    def evalute(self, dl):
        self.model.eval()
        logit_arr = []
        target_arr = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(dl), desc=f"itering in evauation"):
                x = batch["input_ids"]
                y = batch["tags"]
                logit = self.model(x, **batch)
                logit_arr.append(logit)
                target_arr.append(y)
        acc = self._calc_metric(logit_arr, target_arr)
        return acc

    def _calc_metric(self, logit_arr, target_arr):
        logit = torch.cat(logit_arr, dim=0)
        target = torch.cat(target_arr, dim=0)
        target = target.cpu().numpy()
        pred_labels = torch.argmax(logit, dim=1)
        pred_labels = pred_labels.cpu().numpy()
        acc = accuracy_score(target, pred_labels)
        return acc

    def print_metric(self, is_train, epoch, acc, best_f1):
        print("\n")
        stage = "Training" if is_train else "validation"
        print(f"Following is {stage} metrics at epoch {epoch}: ")
        print(f"Accuracy of the model is: {acc}, and best_acc is: {best_f1}")

    def create_optimizer(self, weight_decay, lr):
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        kwargs = {
            "eps": 1e-8,
            "lr": lr
        }
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer

    def get_parameter_names(self, model, forbidden_layer_types):
        res = []
        for name, child in model.named_children():
            res += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        return res

    def create_scheduler(self, optimizer, train_steps, warmup_ratio=None, warmup_steps=None):
        if warmup_steps is None:
            if warmup_ratio is None:
                warmup_ratio = 0.01
            warmup_steps = math.ceil(warmup_ratio * train_steps)
        res = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
        return res

    def get_dataloader(self, f_path, batch_size):
        texts, spans, span_texts, label_list = self.data_prep.prepare(f_path)
        ds = SpanWscDataset(self.tokenizer, texts, spans, label_list)
        dl = dataloader.DataLoader(
                ds, batch_size=batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True, shuffle=True,
                collate_fn=lambda x: ds.collate(
                    x,
                    max_seq_len=self.config.max_seq_len
                )
        )
        return dl


def get_args():
    parser = argparse.ArgumentParser()
    # adding config for the model
    parser.add_argument('--pretrain_name', type=str,
                        help="the file_path or the name of the pretrained model")
    parser.add_argument('--hidden_dropout', type=float, default=0.2,
                        help="dropout ratio for the head")
    parser.add_argument('--num_spans', type=int, default=2,
                        help="indices how many spans to involed in the task")

    # info about dataset
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="maxium seqence length of the task")
    # info about training
    parser.add_argument('--epochs', type=int, default=3,
                        help="epochs to training")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_path', type=str, help='path to the train file')
    parser.add_argument('--valid_path', type=str, help='path to the valid file')
    parser.add_argument('--model_path', type=str, help='where to save the model')
    # parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    # parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    args = parser.parse_args()
    return args


def main(args):
    trainer = WscTrainer(args)
    trainer.train()


if __name__ == "__main__":
    t_args = get_args()
    main(t_args)
