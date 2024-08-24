import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.utils import SE_Block


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation),
        nn.BatchNorm2d(out_planes),  # nn.GroupNorm(1, out_planes)
        nn.ReLU(inplace=True))

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
    def __init__(self, pool_size=4, stride=16):
        super().__init__()

        self.feat1_conv = conv(256, 256)

        self.feat2_conv = conv(256, 256)

        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1 / stride)

        self.channel_attention = SE_Block(256, reduction=4)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        self.ref_kernel = self.prroi_pool(self.feat1_conv(feat1), roi1)  # (64,C,H,W)

    def fuse_feat(self, feat2):
        """fuse features from reference and test branch"""
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: depth-wise neck'''
        feat_corr = depth_corr(self.feat1_conv(feat2), self.ref_kernel)  # (batch,1024,5,5)
        feat_corr = F.interpolate(feat_corr, size=feat2.shape[-2:],
                                  mode='bilinear')  # (batch,64,16,16) # bs, 256 15 15

        feat_corr = self.channel_attention(feat_corr)  # 计算通道注意力特征

        return feat_corr


if __name__=='__main__':
    ref = torch.randn([1, 256, 4, 4])
    feat = torch.randn([1, 256, 18, 18])
    corr = depth_corr(feat, ref)
    print(corr.shape)
