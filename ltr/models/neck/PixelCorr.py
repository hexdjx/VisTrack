
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.neck.utils import *


def pixel_corr(x, kernel):
    """pixel-wise neck"""
    size = kernel.size()
    CORR = []
    for i in range(len(x)):
        ker = kernel[i:i + 1]
        feat = x[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)
        co = F.conv2d(feat, ker.contiguous())
        CORR.append(co)
    corr = torch.cat(CORR, 0)
    return corr


class PixelCorr(nn.Module):

    def __init__(self, pool_size=8, use_post_corr=True, use_NL=True):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1 / 16)
        num_corr_channel = pool_size * pool_size
        self.use_post_corr = use_post_corr
        self.use_NL = use_NL
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
        self.channel_attention = SEModule(num_corr_channel, reduction=4)
        if self.use_NL:
            self.spatial_attention = NONLocalBlock(in_channels=num_corr_channel)

    def forward(self, feat1, feat2, bb1):
        """
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
        """

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

        feat_corr = pixel_corr(feat2, feat_roi1)

        '''channel attention: Squeeze and Excitation'''
        feat_ca = self.channel_attention(feat_corr)  # 计算通道注意力特征

        if self.use_NL:
            '''spatial attention: Non-local'''
            feat_sa = self.spatial_attention(feat_ca)
            return feat_sa

        return feat_ca

    def get_ref_kernel(self, feat1, bb1):
        assert bb1.dim() == 3

        # # Extract first train sample
        # assert len(feat1) == 1  # only support single feature level for now
        # feat1 = feat1[0]
        #
        # sr = 2  # TODO: adaptive crop region according to search region rate
        # self.ref_kernel = feat1[:, :,
        #                   feat1.shape[2] * (sr - 1) // (sr * 2):feat1.shape[2] * (sr + 1) // (sr * 2),
        #                   feat1.shape[3] * (sr - 1) // (sr * 2):feat1.shape[3] * (sr + 1) // (sr * 2)]

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
        """ fuse features from reference and test branch """
        assert len(feat2) == 1
        feat2 = feat2[0]

        # Step1: pixel-wise neck
        feat_corr = pixel_corr(feat2, self.ref_kernel)

        # Step2: channel attention: Squeeze and Excitation
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


