import math
import torch

import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.transformer.transformer as trans
import ltr.models.transformer.filter_predictor as fp
import ltr.models.transformer.heads as heads
import torch.nn.functional as F
from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.utils import conv_bn_relu


class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
              final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
              num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def tompnet101(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
               final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
               num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


# --attention feature fusion-- ########################################################################################
@model_constructor
def fu_tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                 head_feat_norm=True,
                 final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    ################################################################################
    # feature separation and fusion
    head_feature_extractor = clf_features.AWFFatt(out_dim=out_feature_dim, norm_scale=norm_scale)
    ################################################################################

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


class ViT_Backone(nn.Module):
    def __init__(self, img_sz=288):
        super(ViT_Backone, self).__init__()

        self.img_sz = img_sz

        self.backbone_net = backbones.vit_base_patch16(pretrained=True, pretrain_type='mae', search_size=img_sz)

        # self.conv = nn.Conv2d(256,256, kernel_size=1)

        for p in self.backbone_net.parameters():
            p.requires_grad_(False)

        # Init weights
        # for m in self.conv.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()

        x = self.backbone_net(x)  # 768 = 256+512
        b, _, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, self.img_sz // 16, self.img_sz // 16)

        # outputs['layer2'] = F.interpolate(x[:, :256, ...], scale_factor=2, mode='bilinear')
        #
        # outputs['layer3'] = x[:, 256:, ...]
        outputs['layer3'] = x
        return outputs


@model_constructor
def vit_tompnet(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
                final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = ViT_Backone()

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(input_dim=768, feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


class Fuse_Backone(nn.Module):
    def __init__(self, img_sz=288, backbone_pretrained=True,
                 frozen_backbone_layers=['conv1', 'bn1', 'layer1', 'layer2']):
        super(Fuse_Backone, self).__init__()

        self.img_sz = img_sz

        self.vit_net = backbones.vit_large_patch16(pretrained=backbone_pretrained, pretrain_type='mae',
                                                  search_size=img_sz)
        self.cnn_net = backbones.resnet50_v2(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

        for p in self.vit_net.parameters():
            p.requires_grad_(False)

    def forward(self, x, output_layers=['layer3']):
        outputs = OrderedDict()

        x_v = self.vit_net(x)  # 768 = 256+512
        x_c = self.cnn_net(x, output_layers)

        b, _, c = x_v.shape
        x_v = x_v.permute(0, 2, 1).reshape(b, c, self.img_sz // 16, self.img_sz // 16)
        x = torch.cat([x_v, x_c['layer3']], dim=1)
        outputs['layer3'] = x

        return outputs


@model_constructor
def tompnet_swin_base(filter_size=1, head_layer='2', backbone_pretrained=True, head_feat_blocks=0,
                      head_feat_norm=True,
                      final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8,
                      num_encoder_layers=6,
                      num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.swin_base384_flex(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == '2':
        feature_dim = 256
    elif head_layer == '3':
        feature_dim = 512
    elif isinstance(head_layer, list):
        if head_layer[-1] == '2':
            feature_dim = 256
        elif head_layer[-1] == '3':
            feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(input_dim=512, feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def fuse_tompnet(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0, head_feat_norm=True,
                 final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = Fuse_Backone(img_sz=feature_sz * 16, backbone_pretrained=backbone_pretrained,
                                frozen_backbone_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(input_dim=1024 + 1024, feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


#  --Target Probability--  ########################################################################################
class ProbToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_prob, test_prob, train_bb, *args,
                **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        train_prob = F.interpolate(train_prob.reshape(-1, *train_prob.shape[-3:]), size=(18, 18), mode='bilinear')
        test_prob = F.interpolate(test_prob.reshape(-1, *test_prob.shape[-3:]), size=(18, 18), mode='bilinear')

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_prob, test_prob, train_bb, *args,
                                            **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def prob_tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                   head_feat_norm=True,
                   final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                   num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    ########################################################
    # my add classification and probability fusion
    # clf_feature_extractor = conv_bn_relu(4 * feature_dim, out_feature_dim)
    clf_feature_extractor = nn.Conv2d(4 * feature_dim, out_feature_dim, kernel_size=3, padding=1, bias=False)

    prob_feature_extractor = clf_features.prob_encoder_mlp(in_dim=3, hid_dim=16)

    fuse_conv = nn.Sequential(nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=1, padding=0, bias=False),
                              InstanceL2Norm(scale=norm_scale))

    head_feature_extractor = nn.Sequential(clf_feature_extractor, prob_feature_extractor, fuse_conv)
    ########################################################

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Prob_Head(filter_predictor=filter_predictor,
                           feature_extractor=head_feature_extractor,
                           classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ProbToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def prob_tompnet101(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                    head_feat_norm=True,
                    final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                    num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    ########################################################
    # my add classification and probability fusion
    clf_feature_extractor = conv_bn_relu(4 * feature_dim, out_feature_dim, kernel_size=3, padding=1)

    prob_feature_extractor = clf_features.prob_encoder_mlp(in_dim=3, hid_dim=16)

    fuse_conv = nn.Sequential(nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=1, padding=0, bias=False),
                              InstanceL2Norm(scale=norm_scale))

    head_feature_extractor = nn.Sequential(clf_feature_extractor, prob_feature_extractor, fuse_conv)
    ########################################################

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Prob_Head(filter_predictor=filter_predictor,
                           feature_extractor=head_feature_extractor,
                           classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ProbToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


#  --Prompt Tracking--  ########################################################################################
class ProToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_pro, test_pro, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_pro, test_pro, train_bb, *args,
                                            **kwargs)

        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat, target_prob):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat), target_prob)

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def prompt_tompnet101(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                      head_feat_norm=True,
                      final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                      num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True,
                      mlp_dim=3,
                      prompt_mode='prob'):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    ########################################################
    # my add classification and probability fusion
    clf_feature_extractor = conv_bn_relu(4 * feature_dim, out_feature_dim)

    prob_feature_extractor = clf_features.encoder_mlp(in_dim=mlp_dim, hid_dim=16, prompt_mode=prompt_mode)

    fuse_conv = nn.Sequential(nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=1, padding=0, bias=False),
                              InstanceL2Norm(scale=norm_scale))

    head_feature_extractor = nn.Sequential(clf_feature_extractor, prob_feature_extractor, fuse_conv)
    ########################################################

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Prompt_Head(filter_predictor=filter_predictor,
                             feature_extractor=head_feature_extractor,
                             classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ProToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


@model_constructor
def prompt_tompnet50(filter_size=4, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                     head_feat_norm=True,
                     final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8, num_encoder_layers=6,
                     num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True, mlp_dim=3,
                     prompt_mode='prob'):
    # Backbone
    # backbone_net = backbones.resnet50_v2(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    backbone_net = Fuse_Backone(img_sz=feature_sz*16, backbone_pretrained=backbone_pretrained, frozen_backbone_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    ########################################################
    # my add classification and probability fusion
    clf_feature_extractor = conv_bn_relu(8 * feature_dim, out_feature_dim)

    prob_feature_extractor = clf_features.encoder_mlp(in_dim=mlp_dim, hid_dim=16)#, prompt_mode=prompt_mode)
    # prob_feature_extractor = conv_bn_relu(mlp_dim, out_feature_dim, kernel_size=16, stride=16, padding=0)

    fuse_conv = nn.Sequential(nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=3, padding=1, bias=False),
                              InstanceL2Norm(scale=norm_scale))

    head_feature_extractor = nn.Sequential(clf_feature_extractor, prob_feature_extractor, fuse_conv)
    ########################################################

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Prompt_Head(filter_predictor=filter_predictor,
                             feature_extractor=head_feature_extractor,
                             classifier=classifier, bb_regressor=bb_regressor)

    # ToMP network
    net = ProToMPnet(feature_extractor=backbone_net, head=head, head_layer=head_layer)
    return net


# --------------------------------------------------------------------------------------------------------
class SSL_ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, target_embedding, head_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = head

        self.target_embedding = target_embedding

        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, test_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        num_sequences = train_imgs.shape[1]  # n_seq bs c h w

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)

        # target embedding -------------------------------------
        train_target_emb = self.target_embedding(train_feat_head, train_bb.reshape(-1, 4))
        train_target_emb = train_target_emb.reshape(-1, num_sequences, train_target_emb.shape[-1])

        test_target_emb = self.target_embedding(test_feat_head, test_bb.reshape(-1, 4))
        test_target_emb = test_target_emb.reshape(-1, num_sequences, train_target_emb.shape[-1])

        target_emb = [train_target_emb, test_target_emb]
        # --------------------------------------------------------

        # Run head module
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)

        return test_scores, bbox_preds, target_emb

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def ssl_tompnet101(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                   head_feat_norm=True,
                   final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8,
                   num_encoder_layers=6,
                   num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    target_embedding = clf_features.TargetEmbedding()

    # ToMP network
    net = SSL_ToMPnet(feature_extractor=backbone_net, head=head, target_embedding=target_embedding,
                      head_layer=head_layer)
    return net


@model_constructor
def ssl_tompnet50(filter_size=1, head_layer='layer3', backbone_pretrained=True, head_feat_blocks=0,
                  head_feat_norm=True,
                  final_conv=True, out_feature_dim=512, frozen_backbone_layers=(), nhead=8,
                  num_encoder_layers=6,
                  num_decoder_layers=6, dim_feedforward=2048, feature_sz=18, use_test_frame_encoding=True):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if head_layer == 'layer3':
        feature_dim = 256
    elif head_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    head_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                              num_blocks=head_feat_blocks, l2norm=head_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    transformer = trans.Transformer(d_model=out_feature_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)

    filter_predictor = fp.FilterPredictor(transformer, feature_sz=feature_sz,
                                          use_test_frame_encoding=use_test_frame_encoding)

    classifier = heads.LinearFilterClassifier(num_channels=out_feature_dim)

    bb_regressor = heads.DenseBoxRegressor(num_channels=out_feature_dim)

    head = heads.Head(filter_predictor=filter_predictor, feature_extractor=head_feature_extractor,
                      classifier=classifier, bb_regressor=bb_regressor)

    target_embedding = clf_features.TargetEmbedding()

    # ToMP network
    net = SSL_ToMPnet(feature_extractor=backbone_net, head=head, target_embedding=target_embedding,
                      head_layer=head_layer)
    return net
