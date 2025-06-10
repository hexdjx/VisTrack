import torch
import torch.nn.functional as F
import math
from typing import Optional
from torch import nn, Tensor
from .multihead_attention import MultiheadAttention
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.transformer.position_encoding import PositionEmbeddingSine


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=1, key_feature_dim=64):
        super().__init__()

        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=nhead, key_feature_dim=key_feature_dim)

        # FFN_conv = nn.Conv2d()  # do not use feed-forward network

        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model,
                                          num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model,
                                          num_decoder_layers=num_layers)

        self.query_embed_train = nn.Embedding(1, d_model)
        self.query_embed_test = nn.Embedding(1, d_model)

        # self.pos_encoding = PositionEmbeddingSine(num_pos_feats=d_model//2, sine_type='lin_sine',
        #                                           avoid_aliazing=True, max_spatial_resolution=22)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, train_feat, test_feat, train_label, test_label):

        train_token = self.query_embed_train.weight.reshape(1, 1, -1)
        train_label_enc = train_token * train_label

        test_token = self.query_embed_test.weight.reshape(1, 1, -1)
        test_label_enc = test_token * test_label

        # test_pos = self.get_positional_encoding(test_feat) # Nf_te, Ns, C, H, W
        # train_pos = self.get_positional_encoding(train_feat) # Nf_tr, Ns, C, H, W

        ## encoder
        encoded_feat, encoded_memory = self.encoder(train_feat, pos=train_label_enc)

        ## decoder
        decoded_feat = self.decoder(test_feat, memory=encoded_memory, pos=test_label_enc, query_pos=None)

        # for i in range(num_img_train):
        #     _, cur_encoded_feat = self.decoder(train_feat[i, ...].unsqueeze(0), memory=encoded_memory, pos=test_label_enc,
        #                                        query_pos=None)
        #     if i == 0:
        #         encoded_feat = cur_encoded_feat
        #     else:
        #         encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        #
        # for i in range(num_img_test):
        #     _, cur_decoded_feat = self.decoder(test_feat[i, ...].unsqueeze(0), memory=encoded_memory, pos=train_label,
        #                                        query_pos=None)
        #     if i == 0:
        #         decoded_feat = cur_decoded_feat
        #     else:
        #         decoded_feat = torch.cat((decoded_feat, cur_decoded_feat), 0)

        return encoded_feat, decoded_feat


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)
        return src

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, input_shape, pos: Optional[Tensor] = None):
        # self-attention
        src = src + self.self_attn(query=src, key=src, value=self.with_pos_embed(src, pos))
        src = self.instance_norm(src, input_shape)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, pos: Optional[Tensor] = None):
        assert src.dim() == 5, 'Expect 5 dimensional inputs'
        src_shape = src.shape
        num_imgs, batch, dim, h, w = src.shape

        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)

        for layer in self.layers:
            src = layer(src, input_shape=src_shape, pos=pos)

        # [L,B,D] -> [B,D,L]
        src_feat = src.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        src_feat = src_feat.reshape(-1, dim, h, w)
        return src_feat, src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, n_head=8, key_feature_dim=64):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=n_head, key_feature_dim=key_feature_dim)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, tgt, memory, input_shape, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # self-attention
        tgt = tgt + self.self_attn(query=tgt, key=tgt, value=self.with_pos_embed(tgt, pos))
        tgt = self.instance_norm(tgt, input_shape)

        # mask = self.cross_attn(query=tgt, key=memory, value=pos)
        # tgt2 = tgt * mask
        # tgt2 = self.instance_norm(tgt2, input_shape)
        #
        # tgt3 = self.cross_attn(query=tgt, key=memory, value=memory * pos)
        # tgt4 = tgt + tgt3
        # tgt4 = self.instance_norm(tgt4, input_shape)
        #
        # tgt = tgt2 + tgt4

        tgt = tgt + self.cross_attn(query=tgt, key=memory, value=self.with_pos_embed(tgt, pos))
        tgt = self.instance_norm(tgt, input_shape)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert tgt.dim() == 5, 'Expect 5 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim, h, w = tgt.shape

        tgt = tgt.view(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        tgt = tgt.reshape(-1, batch, dim)

        pos = pos.view(num_imgs, batch, 1, -1).permute(0, 3, 1, 2)
        pos = pos.reshape(-1, batch, 1)

        memory = memory.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        memory = memory.reshape(-1, batch, dim)

        for layer in self.layers:
            tgt = layer(tgt, memory, input_shape=tgt_shape, pos=pos, query_pos=query_pos)

        # [L,B,D] -> [B,D,L]
        tgt = tgt.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        tgt = tgt.reshape(-1, dim, h, w)
        # output = self.post2(self.activation(self.post1(output)))
        return tgt


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
