# coding=utf8

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
)


class WscModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(WscModel, self).__init__()
        self.config = config
        self.encoder = self._init_encoder()
        self.cls_num = config.cls_num
        self.head = WscHead(self.encoder.config.hidden_size,
                            self.config.hidden_dropout, self.config.num_spans, self.cls_num)
        self.device = kwargs["device"]

    def _init_encoder(self):
        pretrain_config = AutoConfig.from_pretrained(self.config.pretrain_name)
        model = AutoModel.from_pretrained(
                self.config.pretrain_name,
                from_tf=bool(".ckpt" in self.config.pretrain_name),
                config=pretrain_config
            )
        return model

    def forward(self, x, *wargs, **kwargs):
        """
        {
            "input_ids": input_ids,
            "att_masks": att_masks,
            "tags": tags,
            "spans": spans,
            "spans_width": spans_width,
            "span_masks": span_masks,
        }
        """
        spans = kwargs["spans"]
        attention_mask = kwargs["att_masks"]
        input_ids = x.to(self.device)
        spans = spans.to(self.device)
        # spans_width, spans_mask = spans_width.to(self.device), spans_mask.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros_like(input_ids, device=self.device)

        sequence_out = self._get_enc(
            input_ids, spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        logits = self.head(sequence_out, spans=spans)
        return logits

    def _get_enc(self, input_ids, spans, token_type_ids, attention_mask):
        encoder_output = self.encoder(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_out = encoder_output['last_hidden_state']
        return sequence_out

        # if compute_loss:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.head.num_labels), batch.label_id.view(-1),)
        #     return LogitsAndLossOutput(logits=logits, loss=loss, other=encoder_output.other)
        # else:
        #     return LogitsOutput(logits=logits, other=encoder_output.other)


class WscHead(nn.Module):
    def __init__(self, hidden_dim, hidden_dropout, num_spans, cls_num):
        super(WscHead, self).__init__()
        self.hid_dim = hidden_dim
        self.num_spans = num_spans
        self.cls_num = cls_num
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.num_spans * self.hid_dim, self.cls_num)
        self.span_att_extractor = SelfAttSpanExtractor(self.hid_dim,
                hidden_dropout)

    def forward(self, enc_seq, spans):
        span_emb = self.span_att_extractor(enc_seq, spans)
        span_emb = span_emb.view(-1, self.num_spans * self.hid_dim)
        span_emb = self.dropout(span_emb)
        logits = self.classifier(span_emb)
        return logits


class SelfAttSpanExtractor(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(SelfAttSpanExtractor, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self._global_att_linear = nn.Linear(hid_dim, 1)

    def forward(self, enc_seq, spans):
        att_logits = self._global_att_linear(enc_seq)
        span_begins, span_ends = spans.split(1, dim=-1)
        raw_widths = span_ends - span_begins
        max_width = raw_widths.max().item() + 1
        max_width_range = get_range_vector(max_width,
                get_device_of(enc_seq)).view(1, 1, -1)
        span_mask = (max_width_range <= raw_widths).float()

        raw_span_indices = span_ends - max_width_range
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.relu(raw_span_indices.float()).long()
        flat_indices = flatten_and_batch_shift_indices(span_indices,
                enc_seq.shape[1])
        span_emb = batched_index_select(enc_seq, span_indices, flat_indices)
        span_att_logits = batched_index_select(att_logits, span_indices,
                flat_indices).squeeze(-1)
        span_att_weights = masked_softmax(span_att_logits, span_mask)
        atted_emb= weighted_sum(span_emb, span_att_weights)
        return atted_emb


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices, seq_len):
    offsets = get_range_vector(indices.shape[0], get_device_of(indices)) * seq_len
    for _ in range(len(indices.shape) - 1):
        offsets = offsets.unsqueeze(1)
    res = indices + offsets 
    res = res.view(-1)
    return res


def batched_index_select(target, indices, flattened_indices):
    """
        @param indices: [batch, dim1, ..., dim_n]
        @param target: [batch, seq_len, hid_dim]
        @return: selected_tensor with shape [batch, dim1, ..., dim_n,  hid_dim]
    """
    if flattened_indices is None:
        flattened_indices = flatten_and_batch_shift_indices(indices, target.shape[1])
    flat_target = target.view(-1, target.shape[-1])
    flat_selected = flat_target.index_select(0, flattened_indices)
    selected_shape = list(indices.shape) + [target.shape[-1]]
    res = flat_selected.view(*selected_shape)
    return res


def masked_softmax(logits, mask, dim=-1):
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.float()
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(1)
    masked_logits =  logits * mask
    weight = torch.softmax(masked_logits, dim=dim)
    weight = weight * mask
    weight_sum = weight.sum(dim=dim, keepdim=True)
    return weight / (weight_sum + 1e-13)


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

