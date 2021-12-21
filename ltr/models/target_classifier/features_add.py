import math
import torch
from torch import nn
from ltr.models.layers.normalization import InstanceL2Norm

###################################################################################
# my add
# author Xuedong He 2020.12.17
def conv_bn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


# EnDiMP
class Res_Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Res_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(residual)
        out = self.relu(out)

        return out


class Decode_Feature(nn.Module):
    def __init__(self, norm_scale):
        super(Decode_Feature, self).__init__()

        self.res1 = Res_Bottleneck(1024, 512)
        self.res2 = Res_Bottleneck(512, 256)

        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

        self.final_cov = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.instance_norm = InstanceL2Norm(scale=norm_scale)

        # Init weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.uniform_()
        #         m.bias.data.zero_()

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        # x4 = self.res3(x3)
        x_concat = torch.cat([self.conv1(x1), self.conv2(x2)], dim=1)
        out = self.instance_norm(self.final_cov(x_concat))
        return out


class Feature_Fusion(nn.Module):
    def __init__(self, norm_scale):
        super(Feature_Fusion, self).__init__()

        self.res1 = Res_Bottleneck(1024, 512)
        self.res2 = Res_Bottleneck(512, 256)

        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

        self.final_cov = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.instance_norm = InstanceL2Norm(scale=norm_scale)

        # Init weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.uniform_()
        #         m.bias.data.zero_()

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        # x4 = self.res3(x3)
        x_concat = torch.cat([self.conv1(x1), self.conv2(x2)], dim=1)
        out = self.instance_norm(self.final_cov(x_concat))
        return out


# Adaptively Weighted Feature Fusion
class AWFF(nn.Module):
    def __init__(self, dim=None, inter_dim=1024, out_dim=512, compress_dim=16, norm_scale=1.0):
        super(AWFF, self).__init__()
        if dim is None:
            dim = [256, 512, 1024]
        self.dim = dim

        self.stride_l1 = nn.Sequential(conv_bn_relu(self.dim[0], inter_dim, 1, 1, 0),
                                       conv_bn_relu(inter_dim, inter_dim, 3, 2),
                                       conv_bn_relu(inter_dim, inter_dim, 3, 2),
                                       )
        self.stride_l2 = nn.Sequential(conv_bn_relu(self.dim[1], inter_dim, 1, 1, 0),
                                       conv_bn_relu(inter_dim, inter_dim, 3, 2)
                                       )
        self.stride_l3 = conv_bn_relu(self.dim[2], inter_dim, 1, 1, 0)

        self.conv = nn.Sequential(conv_bn_relu(inter_dim, inter_dim, 3, 1),
                                  nn.Conv2d(inter_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                  InstanceL2Norm(scale=norm_scale)
                                  )

        self.weight_l1 = conv_bn_relu(inter_dim, compress_dim, 1, 1, 0)
        self.weight_l2 = conv_bn_relu(inter_dim, compress_dim, 1, 1, 0)
        self.weight_l3 = conv_bn_relu(inter_dim, compress_dim, 1, 1, 0)
        self.weight_levels = nn.Conv2d(compress_dim * 3, 3, 1, 1, 0)

    def forward(self, x):
        l1, l2, l3 = x['layer1'], x['layer2'], x['layer3']
        # Feature Resizing
        l1_resized = self.stride_l1(l1)
        l2_resized = self.stride_l2(l2)
        l3_resized = self.stride_l3(l3)

        # feature fusion weight learning
        l1_weight = self.weight_l1(l1_resized)
        l2_weight = self.weight_l2(l2_resized)
        l3_weight = self.weight_l3(l3_resized)
        layer_weight = torch.cat((l1_weight, l2_weight, l3_weight), 1)
        layer_weight = F.softmax(self.weight_levels(layer_weight), dim=1)

        # adaptive fusion
        fused_out = l1_resized * layer_weight[:, 0:1, :, :] + \
                    l2_resized * layer_weight[:, 1:2, :, :] + \
                    l3_resized * layer_weight[:, 2:, :, :]

        fused_out = self.conv(fused_out)
        return fused_out


if __name__ == '__main__':
    # x = torch.rand(1, 1024, 16, 16)
    # m = Decode_Feature(1)
    # out = m(x)

    l3 = torch.randn([1, 1024, 22, 22])
    l2 = torch.randn([1, 512, 44, 44])
    l1 = torch.randn([1, 256, 88, 88])
    norm_scale = math.sqrt(1.0 / (512 * 4 * 4))

    asff = AWFF(norm_scale=norm_scale)(l1, l2, l3)

###################################################################################