# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor


class TransFusion(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, mask_temp, pos_temp, src_search, mask_search, pos_search):
        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)

        memory_search = self.encoder(src=src_search,
                                     src_key_padding_mask=mask_search,
                                     pos_src=pos_search)

        hs = self.decoder(src_temp, memory_search,
                          tgt_key_padding_mask=mask_temp,
                          memory_key_padding_mask=mask_search,
                          pos_enc=pos_search, pos_tgt=pos_temp)

        return memory_search, hs


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos_src: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos_src)  # add pos to src

        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_src: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos_src)


class Encoder(nn.Module):

    def __init__(self, EncoderLayer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(EncoderLayer, num_layers)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos_src: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos_src=pos_src)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_tgt: Optional[Tensor] = None):
        # self-attention
        q = k = self.with_pos_embed(tgt, pos_tgt)  # Add object query to the query and key
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_tgt),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_tgt: Optional[Tensor] = None):
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_tgt)


class Decoder(nn.Module):

    def __init__(self, DecoderLayer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(DecoderLayer, num_layers)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_tgt: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_tgt=pos_tgt)
        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

