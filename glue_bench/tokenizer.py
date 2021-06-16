# coding=utf8


import os
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainedTokenizer(object):
    def __init__(self, pretrain_name):
        print(f"tokenizer from: {pretrain_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_name, use_fast=True)

    def tokenize(self, sen):
        words = self.tokenizer.tokenize(sen)
        return words

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, tok_ids):
        return self.tokenizer.convert_ids_to_tokens(tok_ids)

    def get_voc_size(self):
        return len(self.tokenizer.ids_to_tokens)

    def tokenize_detail(self, sen):
        res = self.tokenizer(sen, return_offsets_mapping=True)
        f_res = {
            "tokens": self.tokenizer.convert_ids_to_tokens(res["input_ids"]),
            "token_ids": res["input_ids"],
            "pos_list": res["offset_mapping"],
        }
        return f_res
