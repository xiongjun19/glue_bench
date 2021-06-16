# coding=utf8

import json
from tqdm import tqdm


class WscDataPreper(object):
    def __init__(self):
        self.label_map = {False: 0, True: 1}

    def prepare(self, f_path):
        texts = []
        spans = []
        span_texts = []
        labels = []
        with open(f_path) as in_:
            for line in tqdm(in_):
                line = line.strip()
                if len(line) > 2:
                    obj = json.loads(line)
                    text = obj["text"]
                    target = obj['target']
                    span1_text = target['span1_text']
                    span1_index = target['span1_index']
                    span2_text = target['span2_text']
                    span2_index = target['span2_index']
                    label = self.label_map[bool(obj['label'])]
                    std_span1 = self._get_std_span(text, span1_text, span1_index)
                    std_span2 = self._get_std_span(text, span2_text, span2_index)
                    texts.append(text)
                    spans.append([std_span1, std_span2])
                    span_texts.append([span1_text, span2_text])
                    labels.append(label)
        return texts, spans, span_texts, labels

    def _get_std_span(self, text, span_text, span_index):
        if span_index < 0:
            return [-1, -1]
        word_arr = text.split()
        s = 0
        for i in range(span_index):
            s += len(word_arr[i]) + 1
        return [s,  s + len(span_text)]
        
    def get_cls_num(self):
        return len(self.label_map)

