import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.layers.transform import InterpCat
from ltr.models.utils import conv_bn_relu
from torchvision.ops import RoIAlign


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

# --CAT-- ###########################################################
def prob_encoder_mlp(in_dim=3, hid_dim=16):
    feat_layers = [conv_bn_relu(in_dim, hid_dim, kernel_size=1, padding=0),
                   conv_bn_relu(hid_dim, hid_dim, kernel_size=1, padding=0),
                   conv_bn_relu(hid_dim, in_dim, kernel_size=1, padding=0)]

    return nn.Sequential(*feat_layers)


# --prompt tracking-- ###########################################################
def encoder_mlp(in_dim=3, hid_dim=16):
    """Construct a network block based on the Bottleneck block used in ResNet."""
    feat_layers = [conv_bn_relu(in_dim, hid_dim, kernel_size=1, padding=0),
                   conv_bn_relu(hid_dim, hid_dim, kernel_size=1, padding=0),
                   nn.Conv2d(hid_dim, 1, kernel_size=1),
                   nn.Sigmoid()]

    return nn.Sequential(*feat_layers)


class PE_MLP(nn.Module):
    def __init__(self, inplanes=3, planes=256, norm_scale=1.0,factor=4):
        super(PE_MLP, self).__init__()
        self.clf_conv = conv_bn_relu(factor * planes, planes)
        self.pro_conv = conv_bn_relu(inplanes, planes, kernel_size=16, stride=16, padding=0)
        self.att_mlp = encoder_mlp(256, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, feat, pro):
        feat = self.clf_conv(feat) * self.att_mlp(self.pro_conv(pro))
        return self.final_conv(feat)

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

# patch embedding with attention
class PE_ATT(nn.Module):
    def __init__(self, inplanes=3, planes=256, norm_scale=1.0):
        super(PE_ATT, self).__init__()
        self.clf_conv = conv_bn_relu(4 * planes, planes)  # mlp 2*planes att:planes
        self.pro_conv = conv_bn_relu(inplanes, planes, kernel_size=16, stride=16, padding=0)
        self.c_att = ChannelAtt(in_channel=planes)
        self.s_att = SpatialAtt()

        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * planes, 2 * planes, kernel_size=3, padding=1, bias=False),
            InstanceL2Norm(scale=norm_scale))

    def forward(self, feat, pro):
        c_feat = self.clf_conv(feat) * self.c_att(self.pro_conv(pro))
        s_feat = self.clf_conv(feat) * self.s_att(self.pro_conv(pro))
        return self.final_conv(torch.cat((c_feat, s_feat), dim=1))

# for JDTrack use ----------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wk = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        # Spatial-only Self-Attention
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wk = self.sp_wk(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wk = spatial_wk.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wz = torch.matmul(spatial_wq, spatial_wk)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        return spatial_out, spatial_weight

class TargetEmbeddingRoiAlign(nn.Module):
    def __init__(self, pool_size=4, use_conv=False):
        super().__init__()

        self.roi_align = RoIAlign(pool_size, sampling_ratio=2, spatial_scale=1 / 16)
        self.use_conv = use_conv
        if use_conv:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv = nn.Conv2d(192, 192, kernel_size=1)

    def forward(self, feat, bb):
        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi = torch.cat((batch_index, bb), dim=1)

        feat = self.roi_align(feat, roi)

        if self.use_conv:
            return self.conv(self.pool(feat))
        return feat


class AttFusion(nn.Module):
    def __init__(self, in_channel=768):
        super(AttFusion, self).__init__()
        # compute attention weight
        self.s_att1 = SpatialAttention(channel=in_channel)
        self.s_att2 = SpatialAttention(channel=in_channel)
    def forward(self, x):
        x1, x2 = x[0], x[1]
        x1_s, w1_s = self.s_att1(x1)
        x2_s, w2_s = self.s_att2(x2)

        c_w = torch.cat((w1_s, w2_s), dim=1)
        c_w = F.softmax(c_w, dim=1)
        # adaptive fusion
        x_fusion = x1_s * c_w[:, 0:1, :, :] + x2_s * c_w[:, 1:2, :, :]

        return x_fusion


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

