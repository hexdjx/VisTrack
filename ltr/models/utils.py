import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


# @author: Xuedong He

# -- base conv block-- ############################################################################
def conv_bn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_planes))


def conv_gn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias),
        nn.GroupNorm(1, out_planes),  # instance norm
        nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def conv_gn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias),
        nn.GroupNorm(1, out_planes)]
    return nn.Sequential(*layers)


# --attention block-- ##########################################################################
# Non-local Neural Networks /self-attention，which is a spatial attention block.
class NonLocal_Block(nn.Module):
    def __init__(self, channel):
        super(NonLocal_Block, self).__init__()
        self.inter_channel = channel // 2
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, bias=False)
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, bias=False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # matmul for phi and theta，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1 conv expand channel
        mask = self.conv_mask(mul_theta_phi_g)
        # residual connection
        out = mask + x
        return out


# Squeeze-and-Excitation Networks， which is a channel attention block.
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze operation
        y = self.fc(y).view(b, c, 1, 1)  # FC obtains channel attention weight
        return x * y


# CBAM: Convolutional Block Attention Module
# channel and spatial attention
class CBAM(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(CBAM, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # b, c, w, h = x.size()
        # channel attention
        avg_fc = self.fc(self.avg_pool(x))
        max_fc = self.fc(self.max_pool(x))
        c_out = self.sigmoid(avg_fc + max_fc)
        x_c = x * c_out

        # spatial attention
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_mean, x_max], dim=1)
        x_s = self.sigmoid(self.conv(x_cat))
        out = x_c * x_s

        return out


# separate channel and spatial attention
class ChannelAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))). \
            permute(0, 2, 1).reshape(b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x
        return channel_out


class SpatialAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        return spatial_out


def ScaledDotProductAttention(feat1, feat2, scale=None, pos_mask=None):
    b, c, w, h = feat1.size()
    q = feat1.reshape(b, c, w * h)
    k = v = feat2.reshape(b, c, w * h)
    if pos_mask:
        pos_mask = pos_mask.reshape(b, c, w * h)
        q = q + pos_mask
        k = k + pos_mask

    attention = torch.bmm(q, k.transpose(1, 2))

    if scale:
        attention = attention * scale
    # softmax
    attention = nn.Softmax(dim=-1)(attention)

    # dot-product
    context = torch.bmm(attention, v)

    context = context.reshape(b, c, w, h)

    return context


###################################################################################
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Atrous Spatial Pyramid Pooling
# input: feature map
class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256):
        super(ASPP, self).__init__()
        # conv block with various atrous rate
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        # pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv1x1 = nn.Conv2d(out_channel * 5, out_channel, 1)

    def forward(self, x):
        size = x.shape[2:]
        # atrous conv
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        # pool
        feat_pool = self.avg_pool(x)
        feat_pool = self.conv(feat_pool)
        feat_pool = F.upsample(feat_pool, size=size, mode='bilinear')
        # concat and fuse feature
        x = torch.cat([atrous_block1, atrous_block6, atrous_block12, atrous_block18, feat_pool], dim=1)
        output = self.conv1x1(x)
        return output


# Deformable Convolutional Networks
# if modulation=True, Modulated Defomable Convolution (Deformable ConvNets v2).
class DeformConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, stride=1, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.modulation = modulation

        self.p_conv = nn.Conv2d(in_planes, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=3, bias=False)
        if modulation:
            self.m_conv = nn.Conv2d(in_planes, kernel_size * kernel_size, kernel_size=3, padding=1, bias=False)

    def interpolate(self, x, offset):
        dtype = offset.data.type()
        b, c, h, w = offset.size()
        N = c // 2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        # q_lt, q_rb 插值上下两个整数值
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # pure x index
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # pure y index

        # clip p, avoid index overflow
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        return x_offset

    def forward(self, x):
        # 学习出offset，包括x和y两个方向，注意是每一个channel中的每一个像素都有一个x和y的offset
        offset = self.p_conv(x)
        if self.modulation:  # V2的时候还会额外学习一个权重系数，经过sigmoid拉到0和1之间
            m = torch.sigmoid(self.m_conv(x))
        # 利用offset对x进行插值，获取偏移后的x_offset
        x_offset = self.interpolate(x, offset)
        if self.modulation:  # V2的时候，将权重系数作用到特征图上
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset)
        out = self.conv(x_offset)  # offset作用后，在进行标准的卷积过程
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = p_0_x.view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        _, n, h, w = offset.size()
        N = n // 2

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    def _reshape_x_offset(self, x_offset):
        ks = self.kernel_size
        b, c, h, w, N = x_offset.size()

        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


# Receptive Field Block
class RFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        # 分支0：1X1卷积+3X3卷积
        self.branch0 = nn.Sequential(conv_bn_relu(in_planes, 2 * inter_planes, 1, 1, 0),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, visual, visual, False))
        # 分支1：1X1卷积+3X3卷积+空洞卷积
        self.branch1 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1, 0),
                                     conv_bn_relu(inter_planes, 2 * inter_planes, 3, 1, 1),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, visual + 1, visual + 1,
                                                  False))
        # 分支2：1X1卷积+3X3卷积*3代替5X5卷积+空洞卷积
        self.branch2 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1, 0),
                                     conv_bn_relu(inter_planes, (inter_planes // 2) * 3, 3, 1, 1),
                                     conv_bn_relu((inter_planes // 2) * 3, 2 * inter_planes, 3, 1, 1),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, 2 * visual + 1,
                                                  2 * visual + 1, False))
        self.ConvLinear = conv_bn_relu(6 * inter_planes, out_planes, 1, 1, 0)
        self.shortcut = conv_bn(in_planes, out_planes, 1, 1, 0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # 尺度融合
        out = torch.cat((x0, x1, x2), 1)
        # 1X1卷积
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )
        # cheap操作，注意利用了分组卷积进行通道分离
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(), )

    def forward(self, x):
        x1 = self.primary_conv(x)  # 主要的卷积操作
        x2 = self.cheap_operation(x1)  # cheap变换操作
        out = torch.cat([x1, x2], dim=1)  # 二者cat到一起
        return out[:, :self.oup, :, :]


# Adaptively Spatial Feature Fusion
class ASFF(nn.Module):
    def __init__(self, level, rfb=False):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改
        self.dim = [1024, 512, 256]
        self.inter_dim = self.dim[self.level]
        # 每个层级三者输出通道数需要一致
        if level == 0:
            self.stride_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
            self.expand = conv_bn_relu(self.inter_dim, 1024, 3, 1)
        elif level == 1:
            self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1, 0)
            self.stride_level_2 = conv_bn_relu(self.dim[2], self.inter_dim, 3, 2)
            self.expand = conv_bn_relu(self.inter_dim, 512, 3, 1)
        elif level == 2:
            self.compress_level_0 = conv_bn_relu(self.dim[0], self.inter_dim, 1, 1, 0)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = conv_bn_relu(self.dim[1], self.inter_dim, 1, 1, 0)
            self.expand = conv_bn_relu(self.inter_dim, 256, 3, 1)
        compress_c = 8 if rfb else 16
        self.weight_level_0 = conv_bn_relu(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_1 = conv_bn_relu(self.inter_dim, compress_c, 1, 1, 0)
        self.weight_level_2 = conv_bn_relu(self.inter_dim, compress_c, 1, 1, 0)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

    # 尺度大小 level_0 < level_1 < level_2
    def forward(self, x_level_0, x_level_1, x_level_2):
        # Feature Resizing过程
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        # 融合权重也是来自于网络学习
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v,
                                     level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)  # alpha产生
        # 自适应融合
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)
        return out


# --MDNet--#################################################
# function: sampling required pos/neg samples
def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


class SampleGenerator(object):
    def __init__(self, type_, img_size, trans=1.0, scale=1.0, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # (h, w)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bb, n):

        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))

        # vary aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):

        if overlap_range is None and scale_range is None:
            return self._gen_samples(bbox, n)

        else:
            samples = None
            remain = n
            factor = 2
            while remain > 0 and factor < 16:
                samples_ = self._gen_samples(bbox, remain * factor)

                idx = np.ones(len(samples_), dtype=bool)
                if overlap_range is not None:
                    r = overlap_ratio(samples_, bbox)
                    idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                if scale_range is not None:
                    s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                samples_ = samples_[idx, :]
                samples_ = samples_[:min(remain, len(samples_))]
                if samples is None:
                    samples = samples_
                else:
                    samples = np.concatenate([samples, samples_])
                remain = n - len(samples)
                factor = factor * 2

            return samples

# Large Selective Kernel Network for Remote Sensing Object Detection
class LSKmodule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim//2, 1)
        self.conv1_s = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)

        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv_m (attn)
        return x * attn

############################################################

if __name__ == "__main__":
    # x = torch.randn([1, 512, 16, 16])
    # aspp = ASPP()
    # out = aspp(x)
    # nl = NonLocal_Block(512)(x)
    # se = SE_Block(512)(x)
    # cbam = CBAM(512)(x)
    # df = DeformConv2d(512, 512, modulation=True)(x)
    # g = GhostModule(512, 512)(x)
    # rfb = RFB(512, 512)(x)
    # l1 = torch.randn([1, 1024, 8, 8])
    # l2 = torch.randn([1, 512, 16, 16])
    # l3 = torch.randn([1, 256, 32, 32])

    # asff = ASFF(0)(l1, l2, l3)

    img_path = 'D:/Tracking/Datasets/OTB100/Basketball/img/0001.jpg'
    bbox = np.array([198, 214, 34, 81])
    img = np.array(Image.open(img_path))

    # for example:
    pos_examples = SampleGenerator('gaussian', img.shape[:2], 0.1, 1.3)(
        bbox, 500, [0.7, 1])

    neg_examples = SampleGenerator('gaussian', img.shape[:2], 1.1, 1.3)(
        bbox, 1500, [0, 0.3])

nn.MSELoss()