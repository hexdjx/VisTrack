import math

import torch.nn as nn
from collections import OrderedDict
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor

import ltr.models.target_classifier.transformer as transformer  # trdimp
import timm


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
