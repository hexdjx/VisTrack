import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.verifynet as verify_models
from ltr import model_constructor


class Verify_Net(nn.Module):
    """ verify network module"""

    def __init__(self, feature_extractor, target_embedding, classify_layer, extractor_grad=True):

        super(Verify_Net, self).__init__()

        self.feature_extractor = feature_extractor
        self.target_embedding = target_embedding
        self.classify_layer = classify_layer

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        batch_size = train_imgs.shape[-4]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        train_feat = [feat for feat in train_feat.values()]
        test_feat = [feat for feat in test_feat.values()]

        train_feat_embedding = self.target_embedding(train_feat)
        train_feat_embedding = train_feat_embedding.reshape(-1, batch_size, 256)

        test_feat_embedding = self.target_embedding(test_feat)
        test_feat_embedding = test_feat_embedding.reshape(-1, batch_size, 256)

        return train_feat_embedding, test_feat_embedding

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.classify_layer
        return self.feature_extractor(im, layers)

    def extract_target_embedding(self, backbone_feature):

        backbone_feature = [feat for feat in backbone_feature.values()]

        target_embedding = self.target_embedding(backbone_feature)
        return target_embedding


@model_constructor
def verify_resnet50(backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # feature embedding
    target_embedding = verify_models.Target_Embedding()

    net = Verify_Net(feature_extractor=backbone_net, target_embedding=target_embedding,
                     classify_layer=['layer2', 'layer3'],
                     extractor_grad=False)

    return net
