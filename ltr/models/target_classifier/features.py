import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.transform import InterpCat
from ltr.models.utils import conv_bn_relu, conv_bn, conv_gn_relu, conv_gn, CBAM
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def residual_basic_block(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                         interp_cat=False, final_relu=False, init_pool=False):
    """Construct a network block based on the BasicBlock used in ResNet 18 and 34."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    if init_pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


def residual_basic_block_pool(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0,
                              out_dim=None,
                              pool=True):
    """Construct a network block based on the BasicBlock used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(BasicBlock(feature_dim, odim))
    if final_conv:
        feat_layers.append(nn.Conv2d(feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
    if pool:
        feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))

    return nn.Sequential(*feat_layers)


def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
                        interp_cat=False, final_relu=False, final_pool=False):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    if interp_cat:
        feat_layers.append(InterpCat())
    for i in range(num_blocks):
        planes = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim // 4
        feat_layers.append(Bottleneck(4 * feature_dim, planes))
    if final_conv:
        feat_layers.append(nn.Conv2d(4 * feature_dim, out_dim, kernel_size=3, padding=1, bias=False))
        if final_relu:
            feat_layers.append(nn.ReLU(inplace=True))
        if final_pool:
            feat_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if l2norm:
        feat_layers.append(InstanceL2Norm(scale=norm_scale))
    return nn.Sequential(*feat_layers)


# --EnDiMP-- ###########################################################
# channel reduction by half
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

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x_concat = torch.cat([self.conv1(x1), self.conv2(x2)], dim=1)
        out = self.instance_norm(self.final_cov(x_concat))
        return out


# --FuDiMP-- ###########################################################
# feature fusion with concat
class FetureFusion(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512, norm_scale=1.0):
        super(FetureFusion, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0, bias=True)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=inter_dim, bias=True),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        fused_out = torch.cat([x1, x2], dim=1)
        fused_out = self.final_conv(fused_out)
        return fused_out


# Adaptively Weighted Feature Fusion
class AWFF(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512, norm_scale=1.0):
        super(AWFF, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0, bias=True)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=inter_dim, bias=True),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        # compute weight
        self.weight1 = conv_bn_relu(inter_dim, 8, 1, 1, 0, bias=True)
        self.weight2 = conv_bn_relu(inter_dim, 8, 1, 1, 0, bias=True)
        self.weight_total = nn.Conv2d(8 * 2, 2, 1, 1, 0, bias=True)

        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_dim, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        # feature fusion weight learning
        weight1 = self.weight1(x1)
        weight2 = self.weight2(x2)
        weight_total = torch.cat((weight1, weight2), dim=1)
        weight_total = F.softmax(self.weight_total(weight_total), dim=1)

        # adaptive fusion
        fused_out = x1 * weight_total[:, 0:1, :, :] + x2 * weight_total[:, 1:2, :, :]

        fused_out = self.final_conv(fused_out)
        return fused_out


class Spatial_Attention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        spatial_wv = self.wv(x)  # bs,c//2,h,w
        spatial_wq = self.wq(x)  # bs,c//2,h,w
        spatial_wq = self.avg_pool(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        return spatial_weight


# Adaptively Weighted Feature Fusion with Attention
class AWFFatt(nn.Module):
    def __init__(self, in_dim=1024, out_dim=512, norm_scale=1.0):
        super(AWFFatt, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0, bias=True)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=in_dim // 2, bias=True),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        # compute attention weight
        self.weight1 = Spatial_Attention()
        self.weight2 = Spatial_Attention()

        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_dim, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        # feature fusion weight learning
        weight1 = self.weight1(x1)
        weight2 = self.weight2(x2)
        weight_total = torch.cat((weight1, weight2), dim=1)
        weight_total = F.softmax(weight_total, dim=1)

        # adaptive fusion
        fused_out = x1 * weight_total[:, 0:1, :, :] + x2 * weight_total[:, 1:2, :, :]

        fused_out = self.final_conv(fused_out)
        return fused_out


# --ProbFusion-- ###########################################################
class ResBlock_gn(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock_gn, self).__init__()

        self.conv1 = conv_gn_relu(inplanes, planes, stride=stride)
        self.conv2 = conv_gn(planes, planes)
        if inplanes != planes:
            self.downsample = conv_gn(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            x = self.downsample(x)
            out += x

        out = self.relu(out)
        return out


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = conv_bn_relu(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = conv_bn(planes, planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if inplanes != planes:
            self.downsample = conv_bn(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None

    def forward(self, x):

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            x = self.downsample(x)
            out += x

        out = self.relu(out)

        return out


def prob_encoder_conv(in_dim=3, hid_dim=64, out_dim=512, num_blocks=2):
    """Construct a network block based on the Bottleneck block used in ResNet."""

    feat_layers = [conv_bn_relu(in_dim, hid_dim, kernel_size=3, stride=2, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    for i in range(num_blocks):
        planes = hid_dim * (i + 1)
        feat_layers.append(ResBlock(planes, planes * 2, stride=2))

    feat_layers.append(nn.Conv2d(hid_dim * 4, out_dim, kernel_size=1, bias=False))

    return nn.Sequential(*feat_layers)


def prob_encoder_conv_gn(in_dim=3, hid_dim=64, out_dim=512, num_blocks=2):
    """Construct a network block based on the Bottleneck block used in ResNet."""

    feat_layers = [conv_bn_relu(in_dim, hid_dim, kernel_size=3, stride=2, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    for i in range(num_blocks):
        planes = hid_dim * (i + 1)
        feat_layers.append(ResBlock_gn(planes, planes * 2, stride=2))

    feat_layers.append(nn.Conv2d(hid_dim * 4, out_dim, kernel_size=1, bias=False))

    return nn.Sequential(*feat_layers)


def prob_encoder_mlp(in_dim=3, hid_dim=16):
    """Construct a network block based on the Bottleneck block used in ResNet."""

    feat_layers = [conv_bn_relu(in_dim, hid_dim, kernel_size=1, padding=0),
                   conv_bn_relu(hid_dim, hid_dim, kernel_size=1, padding=0),
                   conv_bn_relu(hid_dim, in_dim, kernel_size=1, padding=0)]

    return nn.Sequential(*feat_layers)


# --ToMP improved--################################################################################
class SpatialAtt(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(1, 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        return self.mlp(x_mean)


class ChannelAtt(nn.Module):
    def __init__(self, in_channel=512):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, in_channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.avg_pool(x))


# Adaptively Attention Feature Fusion
class AttFusion(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, norm_scale=1.0, downsample=False):
        super(AttFusion, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=inter_dim),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        # compute attention weight
        self.c_att1 = ChannelAtt()
        self.c_att2 = ChannelAtt()

        self.s_att1 = SpatialAtt()
        self.s_att2 = SpatialAtt()

        self.conv_fusion = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0)

        self.downsample = downsample
        if self.downsample:
            self.downsample = conv_bn(in_dim, inter_dim, kernel_size=1, padding=0)

        self.final_conv = nn.Sequential(
            nn.Conv2d(inter_dim, out_dim, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        # feature fusion weight learning
        c_w1 = self.c_att1(x1)
        c_w2 = self.c_att2(x2)

        c_w = torch.cat((c_w1, c_w2), dim=1)
        c_w = F.softmax(c_w, dim=1)
        # adaptive fusion
        c_fusion = x1 * c_w[:, 0:1, :, :] + x2 * c_w[:, 1:2, :, :]

        # spatial weight
        s_w1 = self.s_att1(x1)
        s_w2 = self.s_att2(x2)
        s_w = torch.cat((s_w1, s_w2), dim=1)
        s_w = F.softmax(s_w, dim=1)
        s_fusion = x1 * s_w[:, 0:1, :, :] + x2 * s_w[:, 1:2, :, :]

        x_fusion = torch.cat((c_fusion, s_fusion), dim=1)
        fused_out = self.conv_fusion(x_fusion)

        if self.downsample:
            fused_out += self.downsample(x)  # with residual

        fused_out = self.final_conv(fused_out)

        return fused_out


class AttFusionBlock(nn.Module):
    def __init__(self, in_dim=1024, downsample=False):
        super(AttFusionBlock, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=inter_dim),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        # compute attention weight
        self.c_att1 = ChannelAtt()
        self.c_att2 = ChannelAtt()

        self.s_att1 = SpatialAtt()
        self.s_att2 = SpatialAtt()

        self.conv_fusion = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0)

        self.downsample = downsample
        if self.downsample:
            self.downsample = conv_bn(in_dim, inter_dim, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        # feature fusion weight learning
        c_w1 = self.c_att1(x1)
        c_w2 = self.c_att2(x2)

        c_w = torch.cat((c_w1, c_w2), dim=1)
        c_w = F.softmax(c_w, dim=1)
        # adaptive fusion
        c_fusion = x1 * c_w[:, 0:1, :, :] + x2 * c_w[:, 1:2, :, :]

        # spatial weight
        s_w1 = self.s_att1(x1)
        s_w2 = self.s_att2(x2)
        s_w = torch.cat((s_w1, s_w2), dim=1)
        s_w = F.softmax(s_w, dim=1)
        s_fusion = x1 * s_w[:, 0:1, :, :] + x2 * s_w[:, 1:2, :, :]

        x_fusion = torch.cat((c_fusion, s_fusion), dim=1)
        fused_out = self.conv_fusion(x_fusion)

        if self.downsample:
            fused_out += self.downsample(x)  # with residual

        return fused_out


class AWFFattBlcok(nn.Module):
    def __init__(self, in_dim=1024):
        super(AWFFattBlcok, self).__init__()

        inter_dim = in_dim // 2
        self.primary_conv = conv_bn_relu(in_dim, inter_dim, kernel_size=1, padding=0, bias=True)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, 3, 1, 1, groups=in_dim // 2, bias=True),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True))

        # compute attention weight
        self.weight1 = Spatial_Attention()
        self.weight2 = Spatial_Attention()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        # feature fusion weight learning
        weight1 = self.weight1(x1)
        weight2 = self.weight2(x2)
        weight_total = torch.cat((weight1, weight2), dim=1)
        weight_total = F.softmax(weight_total, dim=1)

        # adaptive fusion
        fused_out = x1 * weight_total[:, 0:1, :, :] + x2 * weight_total[:, 1:2, :, :]

        return fused_out


class TargetEmbedding(nn.Module):

    def __init__(self, input_dim=1024):
        super().__init__()

        self.prroi_pool = PrRoIPool2D(3, 3, 1 / 16)

        self.conv = conv_bn_relu(input_dim, input_dim, kernel_size=3, stride=1, padding=0)

        self.target_embedding = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, 512)
        )

    def forward(self, feat, bbox):
        batch_size = feat.size()[0]

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(bbox.device)

        bbox = bbox.reshape(-1, *bbox.shape[-2:])
        num_bb_per_batch = bbox.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        bbox_xyxy = torch.cat((bbox[:, :, 0:2], bbox[:, :, 0:2] + bbox[:, :, 2:4]), dim=2)

        roi = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_bb_per_batch, -1), bbox_xyxy), dim=2)
        roi = roi.reshape(-1, 5).to(bbox_xyxy.device)

        out = self.prroi_pool(feat, roi)
        out = self.conv(out)

        out = self.target_embedding(out.view(batch_size * num_bb_per_batch, -1)).reshape(batch_size, num_bb_per_batch,
                                                                                         -1)

        return out


class Res_CBAM(nn.Module):
    def __init__(self, norm_scale):
        super(Res_CBAM, self).__init__()

        self.cbam = CBAM(1024, 512)

        self.final_cov = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.instance_norm = InstanceL2Norm(scale=norm_scale)

    def forward(self, x):
        out = self.instance_norm(self.final_cov(self.cbam(x)))
        return out

if __name__ == '__main__':
    x = torch.rand(1, 1024, 16, 16)
    # m = Decode_Feature(1)
    # out = m(x)

    # l3 = torch.randn([1, 1024, 22, 22])
    # l2 = torch.randn([1, 512, 44, 44])
    l1 = torch.randn([1, 256, 88, 88])
    # norm_scale = math.sqrt(1.0 / (512 * 4 * 4))
    # asff = AWFF(norm_scale=norm_scale)(l1, l2, l3)

    # im = torch.randn([1, 1, 352, 352])
    # prob_feat = prob_encoder_mlp()
    # a = prob_feat(im)

    # m = nn.Upsample(scale_factor=1/16, mode='bilinear')
    # a = m(im)

    a = AttFusion(downsample=True)(x)
    print(a.shape)
