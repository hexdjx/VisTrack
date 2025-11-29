""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import math
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import json
import cv2
import ltr.models.target_classifier.features as clf_features

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    # mae ViT-B/16-224 pre-trained model
    'vit_base_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch16_224_default': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-L/16-224 pre-trained model
    'vit_large_patch16_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    # mae ViT-H/14-224 pre-trained model
    'vit_huge_patch14_224_mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        input_size=(3, 224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


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


def load_pretrained(model, pretrain_type='default', cfg=None, num_classes=1000, in_chans=3, filter_fn=None,
                    strict=True, use_target_token = False, use_dis_token = True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        print("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    if pretrain_type == 'mae':
        state_dict = state_dict['model']

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        print('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            print('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            print('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if pretrain_type == "mae":
        pass
    elif num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']

    # adjust position encoding  --------------------------------------------------
    pe = state_dict['pos_embed'][:, 1:, :]
    # state_dict['pos_embed'] = pe

    b_pe, hw_pe, c_pe = pe.shape

    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w

    if side_pe != side_num_patches_search:
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                            align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
    else:
        pe_s = pe

    if use_dis_token:
        if side_pe != side_num_patches_template:
            pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                                align_corners=True, mode='bicubic')
            pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
        else:
            pe_t = pe

    cls_pos = state_dict['pos_embed'][:, 0:1, :]

    if use_target_token:
        # -----------------------------------------------------------
        pe_target_2D = nn.functional.interpolate(pe_2D, [4, 4],
                                                 align_corners=True, mode='bicubic')
        pe_target = torch.flatten(pe_target_2D.permute([0, 2, 3, 1]), 1, 2)
        # -----------------------------------------------------------
        if use_dis_token:
            pe_xz = torch.cat((cls_pos, pe_target, pe_s, pe_t), dim=1)
        else:
            pe_xz = torch.cat((cls_pos, pe_target, pe_s), dim=1)

    else:
        pe_xz = torch.cat((cls_pos, pe_s, pe_t), dim=1)

    state_dict['pos_embed'] = pe_xz
    # -------------------------------------------------------------------

    model.load_state_dict(state_dict, strict=False)


# ----------------------------------------------------------------------------------------------
# Use ViT to construct joint extraction and model prediction based on ToMP
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, search_size=384, template_size=192,
                 patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 search_number=1, template_number=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_list = [embed_dim]

        self.num_search = search_number
        self.num_template = template_number

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=search_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # -----------------------------------------------------------------------------------
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)

        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches_template + self.num_patches_search, embed_dim))

        self.box_encoding = MLP([4, embed_dim // 4, embed_dim, embed_dim])
        # -----------------------------------------------------------------------------------

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, images_list, train_label, train_ltrb_target):
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
            fg_token_list.append(self.cls_token.expand(z.shape[0], -1, -1))

        z_feat = torch.cat(z_list, dim=1)
        fg_feat = torch.cat(fg_token_list, dim=1)

        x_list = []
        for i in range(num_search):
            x = search_list[i]
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, self.num_patches_search + 1:, :]
            x_list.append(x)
        x_feat = torch.cat(x_list, dim=1)

        if(x_feat.shape[0]!= z_feat.shape[0]):
            x_feat = torch.cat([x_feat, x_feat], dim=0)
            xz_feat = torch.cat([fg_feat, z_feat, x_feat], dim=1)
        else:
            xz_feat = torch.cat([fg_feat, z_feat, x_feat], dim=1)

        xz = self.pos_drop(xz_feat)

        for blk in self.blocks:  # batch is the first dimension.
            if self.use_checkpoint:
                xz = checkpoint.checkpoint(blk, xz)
            else:
                xz = blk(xz)

        xz = self.norm(xz)  # B,N,C
        return xz

    def forward(self, images_list, train_label, train_ltrb_target):
        h, w = train_label.shape[-2:]

        xz = self.forward_features(images_list, train_label, train_ltrb_target)

        enc_opt = xz[:, -h * w:, :] # b L d # b 324 768
        dec_opt = xz[:, 0:1, :].transpose(1, 2) # b L d # b 1, 768
        return dec_opt.reshape(dec_opt.shape[0], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape([1, enc_opt.shape[0], enc_opt.shape[-1], h, w])


@register_model
def mix_vit_base_patch16(pretrained=False, pretrain_type='default',
                         search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    cfg_type = 'vit_base_patch16_224_' + pretrain_type

    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type,
                        num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def mix_vit_large_patch16(pretrained=False, pretrain_type='default',
                          search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_large_patch16_224_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# including target token based on bounding box -------------------------------------------------------------------------
class VisionTransformer_target(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, search_size=384, template_size=192,
                 patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 search_number=1, template_number=1, use_att_fuse=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_list = [embed_dim]

        self.num_search = search_number
        self.num_template = template_number

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=search_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # target embedding-------------------------------------------------
        # self.target_token = clf_features.TargetEmbeddingPrPool(pool_size=4)
        self.target_token = clf_features.TargetEmbeddingRoiAlign(pool_size=4)

        self.feat_size = search_size // patch_size

        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.use_att_fuse = use_att_fuse

        if self.use_att_fuse is not False:
            self.att_fuse = clf_features.AttFusion(in_channel=embed_dim)

        # -----------------------------------------------------------------------------------
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + 16 + self.num_patches_template + self.num_patches_search, embed_dim))

        self.box_encoding = MLP([4, embed_dim // 4, embed_dim, embed_dim])
        # -----------------------------------------------------------------------------------

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, images_list, train_label, train_ltrb_target, train_bb):
        num_template = self.num_template
        template_list = images_list[0:num_template]
        search_list = images_list[num_template:]
        num_search = len(search_list)

        z_list = []
        fg_token_list = []
        z_target_list =[]
        for i in range(num_template):
            z = template_list[i]
            z = self.patch_embed(z) # b, h*w, c

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
            fg_token_list.append(self.cls_token.expand(z.shape[0], -1, -1))

        # ----------------------------------------------------
        if self.use_att_fuse is not False:
            z_target = self.att_fuse(z_target_list)
            z_target = z_target.flatten(2).transpose(1, 2)  # b 16 c
            z_target = z_target + self.pos_embed[:, 1:16 + 1, :]
        # ----------------------------------------------------

        z_feat = torch.cat(z_list, dim=1)
        fg_feat = torch.cat(fg_token_list, dim=1)

        x_list = []
        for i in range(num_search):
            x = search_list[i]
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, self.num_patches_search + 1 + 16:, :]
            x_list.append(x)
        x_feat = torch.cat(x_list, dim=1)

        xz_feat = torch.cat([fg_feat, z_target, z_feat, x_feat], dim=1)

        xz = self.pos_drop(xz_feat)

        for blk in self.blocks:  # batch is the first dimension.
            if self.use_checkpoint:
                xz = checkpoint.checkpoint(blk, xz)
            else:
                xz = blk(xz)

        xz = self.norm(xz)  # B,N,C
        return xz

    def forward(self, images_list, train_label, train_ltrb_target, train_bb):
        h, w = train_label.shape[-2:]

        xz = self.forward_features(images_list, train_label, train_ltrb_target, train_bb)

        enc_opt = xz[:, -h * w:, :] # b L d # b 324 768
        dec_opt = xz[:, 0:1, :].transpose(1, 2) # b L d # b 1, 768
        return dec_opt.reshape(dec_opt.shape[0], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape([1, enc_opt.shape[0], enc_opt.shape[-1], h, w])


@register_model
def mix_vit_base_patch16_target(pretrained=False, pretrain_type='default',
                                search_size=384, template_size=192, **kwargs):
    patch_size = 16
    model = VisionTransformer_target(
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    cfg_type = 'vit_base_patch16_224_' + pretrain_type

    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type,
                        num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), use_target_token=True)
    return model


if __name__ == '__main__':
    img_sz = 288

    im = cv2.imread('./img/dog.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_sz, img_sz))

    im = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0)

    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    im = im / 255
    im -= mean
    im /= std

    model = mix_vit_base_patch16(pretrained=True, pretrain_type='mae', search_size=img_sz)
    model = model.eval()
    preds = model(im)
    b, _, c = preds.shape
    preds = preds.permute(0, 2, 1)
    preds = preds.reshape(b, c, img_sz // 16, img_sz // 16)
    print(preds.shape)
