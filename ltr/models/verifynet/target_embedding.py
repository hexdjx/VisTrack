import torch.nn as nn
import torch


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class Target_Embedding(nn.Module):

    def __init__(self, input_dim=(512, 1024), use_mlp=False):
        super().__init__()
        # _r for reference, _t for test layer2 layer3
        self.conv1_l2 = conv(input_dim[0], 512, kernel_size=3, stride=1)
        self.conv1_l3 = conv(input_dim[1], 1024, kernel_size=3, stride=1)

        self.conv2_l2 = conv(512, 1024, kernel_size=3, stride=2, padding=1)

        self.conv3_l23 = conv(1024 + 1024, 1024, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_mlp:
            self.feat_embedding = nn.Sequential(
                nn.Linear(input_dim[1], input_dim[1] // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim[1] // 2, input_dim[1] // 4),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim[1] // 4, 256)
            )
        else:
            self.feat_embedding = nn.Linear(1024, 256, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization,
                # which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1).
                # So we use the same initialization here.
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat):
        feat1, feat2 = feat[0], feat[1]
        batch_size = feat1.size()[0]

        feat_l2 = self.conv1_l2(feat1)
        feat_l3 = self.conv1_l3(feat2)

        feat2_l2 = self.conv2_l2(feat_l2)

        feat_l23 = torch.cat((feat2_l2, feat_l3), dim=1)

        feat = self.conv3_l23(feat_l23)
        feat = self.avgpool(feat)
        out = self.feat_embedding(feat.view(batch_size, -1)).reshape(batch_size, -1)

        return out
