
import torch
import torch.nn as nn
import torch.nn.functional as F

from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.utils import SE_Block, NonLocal_Block, conv_bn_relu


def depth_corr(x, kernel):
    """depthwise cross neck
    """
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, pool_size=8, use_post_corr=True, use_NL=True):
        super().__init__()
        '''PrRoIPool2D的第三个参数是当前特征图尺寸相对于原图的比例(下采样率)'''
        '''layer2的stride是8, layer3的stride是16
        当输入分辨率为256x256时, layer3的输出分辨率为16x16, 目标尺寸大约为8x8
        ##### 注意: 如果输入分辨率改变,或者使用的层改变,那么这块的参数需要重新填 #####'''
        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1 / 16)
        num_corr_channel = pool_size * pool_size
        '''newly added'''
        self.adjust_layer = conv_bn_relu(1024, 64)
        self.channel_attention = SE_Block(num_corr_channel, reduction=4)
        self.use_post_corr = use_post_corr
        if use_post_corr:
            self.post_corr = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        self.use_NL = use_NL
        if self.use_NL is True:
            self.spatial_attention = NonLocal_Block(in_channels=num_corr_channel)

    def forward(self, feat1, feat2, bb1):

        assert bb1.dim() == 3

        # Extract first train sample
        if len(feat1) == 1:
            feat1 = feat1[0]  # size为(64,C,H,W)
            feat2 = feat2[0]  # size为(64,C,H,W)
            bb1 = bb1[0, ...]  # (64,4)
        else:
            raise ValueError("目前只支持使用单层特征图")

        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device)  # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1)  # (64,1),(64,4) ---> (64,5)
        feat_roi1 = self.prroi_pool(feat1, roi1)  # (64,C,H,W)

        feat_corr = depth_corr(feat2, feat_roi1)  # (batch,1024,5,5)
        feat_corr = F.interpolate(self.adjust_layer(feat_corr), size=(16, 16), mode='bilinear')  # (batch,64,16,16)

        if self.use_post_corr:
            feat_corr = self.post_corr(feat_corr)

        '''channel attention: Squeeze and Excitation'''
        feat_ca = self.channel_attention(feat_corr)  # 计算通道注意力特征
        if self.use_NL:
            '''spatial attention: Non-local'''
            feat_sa = self.spatial_attention(feat_ca)
            return feat_sa

        return feat_ca

    def get_ref_kernel(self, feat1, bb1):
        assert bb1.dim() == 3

        # Extract first train sample
        if len(feat1) == 1:
            feat1 = feat1[0]  # size为(64,C,H,W)
            bb1 = bb1[0, ...]  # (64,4)
        else:
            raise ValueError("目前只支持使用单层特征图")
        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device)  # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1)  # (64,1),(64,4) ---> (64,5)
        '''注意: feat1和roi1必须都是cuda tensor,不然会计算失败(不报错但是其实cuda已经出现问题了,会影响后续的计算)'''
        self.ref_kernel = self.prroi_pool(feat1, roi1)  # (64,C,H,W)

    def fuse_feat(self, feat2):
        """fuse features from reference and test branch"""
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: depth-wise neck'''
        feat_corr = depth_corr(feat2, self.ref_kernel)  # (batch,1024,5,5)
        feat_corr = F.interpolate(self.adjust_layer(feat_corr), size=(16, 16), mode='bilinear')  # (batch,64,16,16)

        '''Step2: channel attention: Squeeze and Excitation'''
        if self.use_post_corr:
            feat_corr = self.post_corr(feat_corr)

        feat_ca = self.channel_attention(feat_corr)  # 计算通道注意力特征

        '''Step3: spatial attention: Non-local 2D'''
        if self.use_NL is True:
            feat_sa = self.spatial_attention(feat_ca)
            return feat_sa

        # print('not use non-local')
        return feat_ca
