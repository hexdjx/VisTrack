import math

import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
from ltr.models.target_classifier import residual_modules  # for dimpnet50_simple use
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.target_classifier.transformer as transformer  # trdimp
import torch.nn.functional as F
import timm

from ltr.models.neck.DepthCorr import DepthCorr
from ltr.models.neck.PixelCorr import PixelCorr
from ltr.models.neck.MixedCorr import MixedCorr

from ltr.models.head import corner


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    # --oupt-- ###################################
    # for OUPT use
    # def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer,
    #              target_feat_layer=None):
    #############################################
    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer,
                 use_timm=False):
        super().__init__()

        # --oupt-- #################################
        # if target_feat_layer is None:
        #     target_feat_layer = ['layer1', 'layer2', 'layer3', 'layer4']
        #######################################
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        # --oupt-- ####################################
        # self.target_feat_layer = target_feat_layer
        # self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer + self.target_feat_layer)))
        #####################################

        self.use_timm = use_timm

        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        if self.use_timm is not False:
            # for timm models use --------------------------------------------------
            train_feat_maps = self.feature_extractor(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
            train_feat = OrderedDict()
            for i, name in enumerate(['layer2', 'layer3']):
                train_feat[name] = train_feat_maps[i]

            test_feat_maps = self.feature_extractor(test_imgs.reshape(-1, *train_imgs.shape[-3:]))
            test_feat = OrderedDict()
            for i, name in enumerate(['layer2', 'layer3']):
                test_feat[name] = test_feat_maps[i]
            # ---------------------------------------------------------------
        else:
            # Extract backbone features
            train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
            test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers

        if self.use_timm:
            feat_maps = self.feature_extractor(im)
            feat = OrderedDict()
            for i, name in enumerate(['layer2', 'layer3']):
                feat[name] = feat_maps[i]
            return feat

        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # Transformer DiMP--------------------------------------------------------------
    # init_transformer = transformer.Transformer(d_model=512, nhead=1, num_layers=1)
    # #
    # # # The classifier module
    # classifier = target_clf.TransLinearFilter(filter_size=filter_size, filter_initializer=initializer,
    #                                           filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
    #                                           transformer=init_transformer)
    # ----------------------------------------------------------------------------

    # # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def trans_dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                    classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                    clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                    out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                    mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                    score_act='relu', act_param=None, target_mask_act='sigmoid',
                    detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50_v2(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # Transformer DiMP--------------------------------------------------------------
    init_transformer = transformer.Transformer(d_model=512, nhead=8, num_layers=2)
    #
    # # The classifier module
    classifier = target_clf.TransLinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                              filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                              transformer=init_transformer)
    # ----------------------------------------------------------------------------

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def fastvit_dimp(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                 classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                 clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                 out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                 mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                 score_act='relu', act_param=None, target_mask_act='sigmoid',
                 detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    model_name = 'fastvit_sa36'

    file_name = 'fastvit_sa36.apple_in1k'

    file_path = '/home/dell/.cache/huggingface/hub/models--timm--' + file_name + '/pytorch_model.bin'

    pretrained_cfg_overlay = {'file': file_path}
    # load model
    backbone_net = timm.create_model(model_name, features_only=True,
                                     out_indices=[1, 2], pretrained=backbone_pretrained,  # [i + 1 for i in range(3)]
                                     pretrained_cfg_overlay=pretrained_cfg_overlay)

    for p in backbone_net.parameters():
        # if p
        p.requires_grad_(False)

    for p in backbone_net.stages_2.parameters():
        p.requires_grad_(True)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=256,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim, input_dim=256)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(128, 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'], use_timm=True)
    return net


# --Attention Tracking------------------------------------------------------------------------------
@model_constructor
def dimpnet50_att_fusion(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                         classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                         clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                         out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                         mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                         score_act='relu', act_param=None, target_mask_act='sigmoid',
                         detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    ################################################################################
    clf_feature_extractor = clf_features.AttFusion_Ghost(norm_scale=norm_scale)
    ################################################################################

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_cbam(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                   classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                   clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                   out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                   mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                   score_act='relu', act_param=None, target_mask_act='sigmoid',
                   detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    ################################################################################
    clf_feature_extractor = clf_features.Res_CBAM(norm_scale=norm_scale)
    ################################################################################

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_se(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                 classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                 clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                 out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                 mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                 score_act='relu', act_param=None, target_mask_act='sigmoid',
                 detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    ################################################################################
    clf_feature_extractor = clf_features.Res_SE(norm_scale=norm_scale)
    ################################################################################

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


class ViT_Backone(nn.Module):
    def __init__(self, img_sz=352):
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


# import torch
# x = torch.rand(1, 3, 352, 352)
# m = ViT_Backone()
# out = m(x)
# print(out)

@model_constructor
def vit_dimpnet(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                score_act='relu', act_param=None, target_mask_act='sigmoid',
                detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone

    # backbone_net = ViT_Backone()
    backbone_net = backbones.resnet50_v2(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.ViT_AtomIoUNet(pred_inter_dim=256)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer3'])
    return net


@model_constructor
def swin_dimpnet(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                 classification_layer='2', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                 clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                 out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                 mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                 score_act='relu', act_param=None, target_mask_act='sigmoid',
                 detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    # out dim[128, 256, 512, 1024] stage index [0 1 2 3]
    backbone_net = backbones.swin_base384_flex(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == '2':
        feature_dim = 256
    elif classification_layer == '3':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim, input_dim=2 * 256)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(2 * 128, 2 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['1', '2'])
    return net


# -----------------------------------------------------------------------------------------------------------------
class CornerDiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, corr_module, bb_regressor, classification_layer, bb_regressor_layer,
                 use_timm=False):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.corr_module = corr_module
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer

        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_corner = self.get_backbone_bbreg_feat(train_feat)
        test_feat_corner = self.get_backbone_bbreg_feat(test_feat)

        self.corr_module.get_ref_kernel(train_feat_corner, train_bb.view(-1, 4))

        # fuse feature from two branches
        fusion_feat = self.corr_module.fuse_feat(test_feat_corner)

        # Obtain bbox prediction
        corner_pred = self.bb_regressor(fusion_feat)

        return target_scores, corner_pred

    def forward_ref(self, train_feat, train_bb):
        # get reference feature
        self.corr_module.get_ref_kernel(train_feat, train_bb)

    def forward_test(self, test_feat):
        """ Forward pass of test branch. size of test_imgs is (1, batch, 3, 256, 256)"""
        # fuse feature from two branches
        fusion_feat = self.corr_module.fuse_feat(test_feat)

        # Obtain bbox prediction
        corner_pred = self.bb_regressor(fusion_feat)

        return corner_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers

        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def corner_dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                     classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                     clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                     out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                     mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                     score_act='relu', act_param=None, target_mask_act='sigmoid',
                     detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet50_v2(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # corr
    corr_module = MixedCorr(input_dim=1024, pool_size=3)

    # Bounding box regressor
    corner_head = corner.Corner_Predictor(output_sz=22)

    # DiMP network
    net = CornerDiMPnet(feature_extractor=backbone_net, classifier=classifier, corr_module=corr_module,
                        bb_regressor=corner_head,
                        classification_layer=classification_layer, bb_regressor_layer=['layer3'])
    return net
