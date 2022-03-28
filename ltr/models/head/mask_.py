import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class Mask_Predictor_coarse(nn.Module):
    """ Mask Predictor module"""

    def __init__(self, inplanes=64, channel=256):
        super(Mask_Predictor_coarse, self).__init__()
        self.conv1 = conv(inplanes, channel)
        self.conv2 = conv(channel, channel)
        self.conv3 = conv(channel, channel)
        self.conv4 = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                   nn.Sigmoid())

    def forward(self, x):
        """ Forward pass with input x. """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = nn.functional.interpolate(x, scale_factor=16, mode='bilinear')  # 由于下采样了16倍,这里再上采样16倍以与输入分辨率对齐
        return output


# more fine mask predictor, fuse backbone features (refer to SiamMask)
class Mask_Predictor(nn.Module):
    def __init__(self):
        super(Mask_Predictor, self).__init__()
        self.v0 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.v1 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.h0 = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, corr_feat, Lfeat):
        """
        corr_feat是经过correlation和Non-local处理后的特征, Lfeat是backbone提取出的底层特征
        corr_feat: (batch,64,16,16)
        Lfeat: (batch,64,128,128), (batch,256,64,64), (batch,512,32,32)
        """
        # (b,32,32,32)+(b,32,32,32) --> (b,16,32,32)
        out = self.post0(F.interpolate(self.h2(corr_feat), size=(32, 32), mode='bilinear') + self.v2(Lfeat[2]))
        # (b,16,64,64)+(b,16,64,64) --> (b,4,64,64)
        out = self.post1(F.interpolate(self.h1(out), size=(64, 64), mode='bilinear') + self.v1(Lfeat[1]))
        # (b,4,128,128)+(b,4,128,128) --> (b,1,128,128)
        out = self.post2(F.interpolate(self.h0(out), size=(128, 128), mode='bilinear') + self.v0(Lfeat[0]))
        # (b,1,256,256)
        out = torch.sigmoid(F.interpolate(out, size=(256, 256), mode='bilinear'))
        return out
