# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
from timm.models import create_model

import math
import ltr.models.target_classifier.features as clf_features

# from mamba_ssm.modules.mamba_simple import Mamba
from ltr.models.mamba_ssm.modules.mamba_simple import Mamba

import random

from ltr.models.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from ltr.models.backbone.rope import VisionRotaryEmbeddingFast

def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        # import ipdb; ipdb.set_trace()
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        d_state=16,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    # import ipdb; ipdb.set_trace()
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, bimamba_type=bimamba_type,
                        if_divide_out=if_divide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self,
                 search_size=224, template_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 d_state=16,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 search_number=1, template_number=1,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.num_search = search_number
        self.num_template = template_number

        self.patch_embed = PatchEmbed(
            img_size=search_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)

        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

#-----------------------------------------------------------------------------------------
        self.box_encoding = MLP([4, embed_dim // 4, embed_dim, embed_dim])
# ------------------------------------------------------------------------------------------------------

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens + self.num_patches_template + self.num_patches_search, embed_dim))

            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = search_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )

        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)

        # self.head.apply(segm_init_weights)

        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, images_list, train_label, train_ltrb_target, inference_params=None,
                         if_random_cls_token_position=False, if_random_token_rank=False):

        num_template = self.num_template
        template_list = images_list[0:num_template]
        search_list = images_list[num_template:]
        num_search = len(search_list)

        z_list = []
        fg_token_list = []
        for i in range(num_template):
            z = template_list[i]
            z = self.patch_embed(z)

            train_label_seq = train_label[i].flatten(1).unsqueeze(2)  # Nf_tr*H*W,Ns,1
            train_label_enc = self.cls_token * train_label_seq

            train_ltrb_target_seq_T = train_ltrb_target[i].flatten(2)
            train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(0, 2, 1)

            z = z + self.pos_embed[:, 1:self.num_patches_search + 1, :] + train_label_enc + train_ltrb_target_enc

            z_list.append(z)

        z_feat = torch.cat(z_list, dim=1)

        fg_feat = self.cls_token.expand(z_feat.shape[0], -1, -1)

        x_list = []
        for i in range(num_search):
            x_s = search_list[i]
            x_s = self.patch_embed(x_s)
            x_s = x_s + self.pos_embed[:, self.num_patches_search + 1:, :]
            x_list.append(x_s)
        x_feat = torch.cat(x_list, dim=1)

        if (x_feat.shape[0] != z_feat.shape[0]):
            x_feat = torch.cat([x_feat, x_feat], dim=0)
            xz_feat = torch.cat([fg_feat, z_feat, x_feat], dim=1)
        else:
            xz_feat = torch.cat([fg_feat, z_feat, x_feat], dim=1)

        x = self.pos_drop(xz_feat)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]),
                    inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, images_list, train_label, train_ltrb_target):
        h, w = train_label.shape[-2:]

        xz = self.forward_features(images_list, train_label, train_ltrb_target)

        enc_opt = xz[:, -h * w:, :]  # b L d # b 324 768
        dec_opt = xz[:, 0:1, :].transpose(1, 2)  # b L d # b 1, 768
        return dec_opt.reshape(dec_opt.shape[0], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(
            [1, enc_opt.shape[0], enc_opt.shape[-1], h, w])


class VisionMamba_target(nn.Module):
    def __init__(self,
                 search_size=224, template_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 d_state=16,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = True,
                 initializer_cfg=None,
                 fused_add_norm=True,
                 residual_in_fp32=True,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=True,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="v2",
                 if_cls_token=True,
                 if_divide_out=True,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=True,
                 search_number=1, template_number=1, use_att_fuse=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.num_search = search_number
        self.num_template = template_number

        self.patch_embed = PatchEmbed(
            img_size=search_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)

        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

#-----------------------------------------------------------------------------------------
        self.box_encoding = MLP([4, embed_dim // 4, embed_dim, embed_dim])

        # target embedding-------------------------------------------------
        self.target_token = clf_features.TargetEmbeddingRoiAlign(pool_size=4)

        self.use_att_fuse = use_att_fuse

        if self.use_att_fuse is not False:
            self.att_fuse = clf_features.AttFusion(in_channel=embed_dim)

        self.feat_size = search_size // patch_size

# ------------------------------------------------------------------------------------------------------

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens + 16 + self.num_patches_template + self.num_patches_search, embed_dim))

            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = search_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )

        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)

        # self.head.apply(segm_init_weights)

        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, images_list, train_label, train_ltrb_target, train_bb, inference_params=None,
                         if_random_cls_token_position=False, if_random_token_rank=False):

        num_template = self.num_template
        template_list = images_list[0:num_template]
        search_list = images_list[num_template:]
        num_search = len(search_list)

        z_list = []
        z_target_list = []
        for i in range(num_template):
            z = template_list[i]
            z = self.patch_embed(z)

            if self.use_att_fuse is not False:
                z_feat = z.transpose(1, 2).reshape(-1, self.embed_dim, self.feat_size, self.feat_size)
                z_target = self.target_token(z_feat, train_bb[i, ...].reshape(-1, 4))
                z_target_list.append(z_target)
            else:
                # target embedding-------------------------------------------------
                if i == 0:
                    z_feat = z.transpose(1, 2).reshape(-1, self.embed_dim, self.feat_size, self.feat_size)
                    z_target = self.target_token(z_feat, train_bb[0, ...].reshape(-1, 4))
                    z_target = z_target.flatten(2).transpose(1, 2)  # b 16 c
                    z_target = z_target + self.pos_embed[:, 1:16 + 1, :]
            # ----------------------------------------------------

            train_label_seq = train_label[i].flatten(1).unsqueeze(2)  # Nf_tr*H*W,Ns,1
            train_label_enc = self.cls_token * train_label_seq

            train_ltrb_target_seq_T = train_ltrb_target[i].flatten(2)
            train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(0, 2, 1)

            z = z + self.pos_embed[:, 1+16:self.num_patches_search + 1+16, :] + train_label_enc + train_ltrb_target_enc

            z_list.append(z)

        # ----------------------------------------------------
        if self.use_att_fuse is not False:
            z_target = self.att_fuse(z_target_list)
            z_target = z_target.flatten(2).transpose(1, 2)  # b 16 c
            z_target = z_target + self.pos_embed[:, 1:16 + 1, :]
        # ----------------------------------------------------

        z_feat = torch.cat(z_list, dim=1)

        fg_feat = self.cls_token.expand(z_feat.shape[0], -1, -1)

        x_list = []
        for i in range(num_search):
            x_s = search_list[i]
            x_s = self.patch_embed(x_s)
            x_s = x_s + self.pos_embed[:, self.num_patches_search + 1 + 16:, :]
            x_list.append(x_s)
        x_feat = torch.cat(x_list, dim=1)


        xz_feat = torch.cat([fg_feat, z_target, z_feat, x_feat], dim=1)

        x = self.pos_drop(xz_feat)

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]),
                    inference_params=inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, images_list, train_label, train_ltrb_target,train_bb):
        h, w = train_label.shape[-2:]

        xz = self.forward_features(images_list, train_label, train_ltrb_target,train_bb)

        enc_opt = xz[:, -h * w:, :]  # b L d # b 324 768
        dec_opt = xz[:, 0:1, :].transpose(1, 2)  # b L d # b 1, 768
        return dec_opt.reshape(dec_opt.shape[0], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(
            [1, enc_opt.shape[0], enc_opt.shape[-1], h, w])


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                             **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def mix_vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, if_rope=False, final_pool_type='all', if_rope_residual=False, bimamba_type="v2",
        if_cls_token=False, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="to.do",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])

        # checkpoint = torch.load('/media/ad/D0774FE583F6C969/networks/pretrained_models/hustvlVim-small-midclstok'
        #                         '+/vim_s_midclstok_80p5acc.pth', map_location="cpu")
        #
        # model.load_state_dict(checkpoint["model"], strict=False)

    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), strict=False)

    return model


@register_model
def vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False,
                                                                                              **kwargs):
    model = VisionMamba(
        patch_size=16, stride=8, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def mix_vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="to.do",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), strict=False)
    return model


@register_model
def mix_target_vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=False, **kwargs):
    model = VisionMamba_target(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        final_pool_type='all', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2",
        if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="to.do",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    if pretrained:
        load_pretrained_target(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), strict=False)
    return model

def load_pretrained(model, pretrain_type='default', num_classes=1000, in_chans=3,
                    strict=True):
    state_dict = torch.load(
        '/media/ad/D0774FE583F6C969/networks/pretrained_models/vim_b_midclstok_81p9acc.pth', # vim_s_midclstok_80p5acc vim_b_midclstok_81p9acc
        map_location="cpu")

    state_dict = state_dict['model']

    # adjust position encoding  --------------------------------------------------
    pe = state_dict['pos_embed'][:, 1:, :]
    # state_dict['pos_embed'] = pe

    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.patch_embed.num_patches))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))

    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w

    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                            align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_s = pe

    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                            align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_t = pe

    cls_pos = state_dict['pos_embed'][:, 0:1, :]

    pe_xz = torch.cat((cls_pos, pe_s, pe_t), dim=1)

    state_dict['pos_embed'] = pe_xz
    # -------------------------------------------------------------------

    model.load_state_dict(state_dict, strict=strict)


def load_pretrained_target(model, pretrain_type='default', num_classes=1000, in_chans=3,
                    strict=True):
    state_dict = torch.load(
        '/media/ad/D0774FE583F6C969/networks/pretrained_models/vim_b_midclstok_81p9acc.pth', # vim_s_midclstok_80p5acc vim_b_midclstok_81p9acc
        map_location="cpu")

    state_dict = state_dict['model']

    # adjust position encoding  --------------------------------------------------
    pe = state_dict['pos_embed'][:, 1:, :]
    # state_dict['pos_embed'] = pe

    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.patch_embed.num_patches))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))

    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w


    # -----------------------------------------------------------
    pe_target_2D = nn.functional.interpolate(pe_2D, [4, 4],
                                        align_corners=True, mode='bicubic')
    pe_target = torch.flatten(pe_target_2D.permute([0, 2, 3, 1]), 1, 2)
    # -----------------------------------------------------------

    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                            align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_s = pe

    if side_pe != side_num_patches_template:
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                            align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_t = pe

    cls_pos = state_dict['pos_embed'][:, 0:1, :]

    pe_xz = torch.cat((cls_pos, pe_target, pe_s, pe_t), dim=1)

    state_dict['pos_embed'] = pe_xz
    # -------------------------------------------------------------------

    model.load_state_dict(state_dict, strict=strict)

if __name__ == '__main__':
    # backbone_net = create_model(model_name='vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', pretrained=True, img_size=288)
    import ltr.models.backbone as backbones

    backbone_net = backbones.mix_vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
        pretrained=True, img_size=18 * 16)

    net = backbone_net.cuda()
    img = torch.Tensor(1, 3, 288, 288).cuda()

    out = net(img)
    print(out.shape)