# coding=utf8


import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class DatasetMixin(Dataset):
    def __init__(self):
        super(DatasetMixin, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()
        if isinstance(index, slice):
            begin, end, step = index.indices(len(self))
            return [self.get_example(i) for i in range(begin, end, step)]
        if isinstance(index, list):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def get_example(self, i):
        raise NotImplementedError


class SpanWscDataset(DatasetMixin):
    def __init__(self, tokenizer, texts, spans_list, tags_list=None):
        super(SpanWscDataset, self).__init__()
        self.tokenizer = tokenizer
        self.texts = texts
        self.spans_list = spans_list
        self.tags_list = tags_list

    def get_example(self, i):
        text = self.texts[i]
        spans = self.spans_list[i]
        tags = [0] * len(spans)
        if self.tags_list is not None:
            tags = self.tags_list[i]
        token_info = self.tokenizer.tokenize_detail(text)
        token_ids = token_info["token_ids"]
        pos_list = token_info["pos_list"]
        begin_dict, end_dict = self._pos_list2dict(pos_list)
        aligned_spans = [self._span_align_to_token_pos(begin_dict, end_dict, span) for span in spans]
        spans_width = [e - s for s, e in aligned_spans]
        return {"input_ids": token_ids, "tags": tags, "spans": aligned_spans,
                "spans_width": spans_width, "pos_list": pos_list}

    def _span_align_to_token_pos(self, begin_dict,  end_dict, span):
        s, e = span
        s_idx = begin_dict.get(s, -1)
        e_idx = end_dict.get(e, -1)
        return s_idx, e_idx

    def _pos_list2dict(self, pos_list):
        begin_dict = dict()
        end_dict = dict()
        for i, (begin, end) in enumerate(pos_list):
            if end - begin > 0:
                begin_dict[begin] = i
                end_dict[end] = i
        return begin_dict, end_dict

    def __len__(self):
        return len(self.texts)

    @classmethod
    def build_dataset(cls, conf, tokenizer, texts, spans_list, tags_list=None):
        return cls(tokenizer, texts, spans_list, tags_list)

    @classmethod
    def collate(cls, batch, max_seq_len=512, padding_type="fix"):
        _dict = list_dict2dict(batch)
        if padding_type == "dynamic":
            return cls._dynamic_pad(_dict, max_seq_len)
        return cls._fix_pad(_dict, max_seq_len)

    @classmethod
    def _dynamic_pad(cls, _dict, upper_bound=512):
        input_ids_arr = _dict["input_ids"]
        seq_len_arr = [len(input_ids) for input_ids in input_ids_arr]
        max_seq_len = max(seq_len_arr)
        upper_bound = min(max_seq_len, upper_bound)
        return cls._pad(_dict, upper_bound)

    @classmethod
    def _fix_pad(cls, _dict, upper_bound=512):
        return cls._pad(_dict, upper_bound)

    @classmethod
    def _pad(cls, _dict, upper_bound):
        # {"input_ids": token_ids, "tags": tags, "spans": aligned_spans, "spans_width": spans_width}
        input_ids, att_masks = cls._pad_input_ids(_dict["input_ids"], upper_bound)
        tags, spans, spans_width, span_masks = cls._pad_spans(_dict["tags"], _dict["spans"], _dict["spans_width"])
        return {
            "input_ids": torch.LongTensor(input_ids),
            "att_masks": torch.LongTensor(att_masks),
            "tags": torch.LongTensor(tags),
            "spans": torch.LongTensor(spans),
            "spans_width": torch.LongTensor(spans_width),
            "span_masks": torch.LongTensor(span_masks),
            "pos_list": np.array(_dict["pos_list"]),
        }

    @classmethod
    def _pad_input_ids(cls, input_ids_arr, upper_bound):
        res = []
        mask_res = []
        seq_len_arr = [len(input_ids) for input_ids in input_ids_arr]
        for input_ids, seq_len in zip(input_ids_arr, seq_len_arr):
            cur_mask = [1] * seq_len
            if seq_len >= upper_bound:
                res.append(input_ids[:upper_bound])
                mask_res.append(cur_mask[:upper_bound])
            else:
                res.append(input_ids + [0] * (upper_bound - seq_len))
                mask_res.append(cur_mask + [0] * (upper_bound - seq_len))
        return np.array(res), np.array(mask_res)

    @classmethod
    def _pad_spans(cls, tags_arr, spans_arr, spans_width):
        res_tags = tags_arr 
        res_spans = []
        res_spans_width = []
        res_mask = []
        tag_num_arr = [len(tags) for tags in tags_arr]
        max_tag = max(tag_num_arr)
        for tags, spans, width_arr, tag_num in zip(tags_arr, spans_arr, spans_width, tag_num_arr):
            tag_mask = [1] * tag_num
            if tag_num < max_tag:
                res_mask.append(tag_mask + [0] * (max_tag - tag_num))
                res_spans_width.append(width_arr + [0] * (max_tag - tag_num))
                tmp_spans = np.zeros([max_tag, 2], np.long)
                tmp_spans[:tag_num, :] = spans
                res_spans.append(tmp_spans)
            else:
                res_spans.append(spans)
                res_spans_width.append(width_arr)
                res_mask.append(tag_mask)
        return np.array(res_tags, dtype=np.long), np.array(res_spans, dtype=np.long), res_spans_width, res_mask


def list_dict2dict(dict_list):
    """
    将一组dict依据其key，进行合并
    :param dict_list:
    :return:
    """
    res = dict()
    for obj in dict_list:
        for key, val in obj.items():
            if key not in res:
                res[key] = list()
            res[key].append(val)
    return res
