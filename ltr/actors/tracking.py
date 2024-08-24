from . import BaseActor
import torch
import numpy as np


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class ToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()

        return loss, stats


# --Target Probability -----------------------------------------------------------------
class ProbDiMPActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_prob=data['train_target_prob'],
                                            test_prob=data['test_target_prob'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class ProbToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_prob=data['train_target_prob'],
                                             test_prob=data['test_target_prob'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()

        return loss, stats


# --Prompt tracking-----------------------------------------------------------------
class PromptDiMPActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None, prompt_mode='prob'):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.prompt_mode = prompt_mode

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        if self.prompt_mode == 'prob':
            # Run network
            target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_pro=data['train_prob'],
                                                test_pro=data['test_prob'],
                                                train_bb=data['train_anno'],
                                                test_proposals=data['test_proposals'])
        if self.prompt_mode == 'mask':
            # Run network
            target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_pro=data['train_mask'],
                                                test_pro=data['test_mask'],
                                                train_bb=data['train_anno'],
                                                test_proposals=data['test_proposals'])

        if self.prompt_mode == 'label':
            # Run network
            target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                                test_imgs=data['test_images'],
                                                train_pro=data['train_gauss_label'].unsqueeze(-3),
                                                test_pro=data['test_gauss_label'].unsqueeze(-3),
                                                train_bb=data['train_anno'],
                                                test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class PromptToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds = self.net(train_imgs=data['train_images'],
                                             test_imgs=data['test_images'],
                                             train_pro=data['train_pro'],
                                             test_pro=data['test_pro'],
                                             train_bb=data['train_anno'],
                                             train_label=data['train_label'],
                                             train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),
                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()

        return loss, stats


# self supervised learning ---------------------------------------------------------------
class SSL_DiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores, target_emb = self.net(train_imgs=data['train_images'],
                                                        test_imgs=data['test_images'],
                                                        train_bb=data['train_anno'],
                                                        test_bb=data['test_anno'],
                                                        test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # -------------------------------------------------------------------------------
        # Compute ssl loss
        target_emb_loss = self.objective['target_emb_loss'](target_emb[0], target_emb[1])
        loss_target_emb = self.loss_weight['target_emb_loss'] * target_emb_loss
        # -------------------------------------------------------------------------------

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + loss_target_emb

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item(),

                 'Loss/target_emb': target_emb_loss.item(),
                 'Loss/loss_target_emb': loss_target_emb.item(),
                 }

        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class SSL_ToMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def compute_iou_at_max_score_pos(self, scores, ltrb_gth, ltrb_pred):
        if ltrb_pred.dim() == 4:
            ltrb_pred = ltrb_pred.unsqueeze(0)

        n = scores.shape[1]
        ids = scores.reshape(1, n, -1).max(dim=2)[1]
        g = ltrb_gth.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)
        p = ltrb_pred.flatten(3)[0, torch.arange(0, n), :, ids].view(1, n, 4, 1, 1)

        _, ious_pred_center = self.objective['giou'](p, g)  # nf x ns x x 4 x h x w
        ious_pred_center[g.view(n, 4).min(dim=1)[0] < 0] = 0

        return ious_pred_center

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bbox_preds, target_embs = self.net(train_imgs=data['train_images'],
                                                          test_imgs=data['test_images'],
                                                          train_bb=data['train_anno'],
                                                          test_bb=data['test_anno'],
                                                          train_label=data['train_label'],
                                                          train_ltrb_target=data['train_ltrb_target'])

        loss_giou, ious = self.objective['giou'](bbox_preds, data['test_ltrb_target'], data['test_sample_region'])

        # Classification losses for the different optimization iterations
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        # -------------------------------------------------------------------------------
        # Compute ssl loss
        target_emb = self.objective['target_emb'](target_embs[0][0, ...], target_embs[1][0, ...]) + self.objective[
            'target_emb'](target_embs[0][1, ...], target_embs[1][0, ...])
        # -------------------------------------------------------------------------------

        loss = self.loss_weight['giou'] * loss_giou + self.loss_weight['test_clf'] * clf_loss_test + \
               self.loss_weight['target_emb'] * target_emb

        if torch.isnan(loss):
            raise ValueError('NaN detected in loss')

        ious_pred_center = self.compute_iou_at_max_score_pos(target_scores, data['test_ltrb_target'], bbox_preds)

        stats = {'Loss/total': loss.item(),
                 'Loss/GIoU': loss_giou.item(),
                 'Loss/weighted_GIoU': self.loss_weight['giou'] * loss_giou.item(),
                 'Loss/clf_loss_test': clf_loss_test.item(),
                 'Loss/weighted_clf_loss_test': self.loss_weight['test_clf'] * clf_loss_test.item(),

                 'Loss/target_emb': target_emb.item(),

                 'mIoU': ious.mean().item(),
                 'maxIoU': ious.max().item(),
                 'minIoU': ious.min().item(),
                 'mIoU_pred_center': ious_pred_center.mean().item()}

        if ious.max().item() > 0:
            stats['stdIoU'] = ious[ious > 0].std().item()

        return loss, stats


# TranT ---------------------------------------------------------------
class TranstActor(BaseActor):
    """ Actor for training the TransT"""

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        import time
        outputs = self.net(data['search_images'], data['template_images'])

        # generate labels
        targets = []
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w = data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1, -1])
            target_origin[0][0] += target_origin[0][2] / 2
            target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            target_origin[0][1] /= h
            target_origin[0][2] /= w
            target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats


# Trans dimp
class TransDiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_label=data['train_label'],
                                            test_label=data['test_label'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats


class CornerKLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""

    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, corner_pred = self.net(train_imgs=data['train_images'],
                                              test_imgs=data['test_images'],
                                              train_bb=data['train_anno'])

        # Reshape bb reg variables
        # is_valid = data['test_anno'][:, :, 0] < 99999.0
        # bb_scores = bb_scores[is_valid, :]
        # proposal_density = data['proposal_density'][is_valid, :]
        # gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        # bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        # get groundtruth
        bbox_gt = data['test_anno'].view(-1, 4)  # squeeze(0)  # (x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)

        giou_loss = self.objective['giou'](corner_pred, bbox_gt_xyxy)
        l1_loss = self.objective['l1'](corner_pred, bbox_gt_xyxy)
        loss_bb_corner = 2.0 * giou_loss + 5.0 * l1_loss

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in
                               target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2, -1)) for s in
                             target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_corner + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
               loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/l1_loss': l1_loss.item(),
                 'Loss/giou_loss': giou_loss.item(),
                 'Loss/loss_bb_ce': loss_bb_corner.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats
