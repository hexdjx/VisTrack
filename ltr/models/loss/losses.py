import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BalancedBCELoss', 'FocalLoss', 'SmoothL1Loss']


class BalancedBCELoss(nn.Module):

    def __init__(self, neg_weight=1.):
        super(BalancedBCELoss, self).__init__()
        self.neg_weight = neg_weight

    def balanced_bce_loss(self, input, target):
        target = target.type_as(input)
        pos_mask, neg_mask = target.eq(1), target.eq(0)
        pos_num = pos_mask.sum().float().clamp(1e-12)
        neg_num = neg_mask.sum().float().clamp(1e-12)
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1. / pos_num
        weight[neg_mask] = (1. / neg_num) * self.neg_weight
        weight /= weight.mean()
        return F.binary_cross_entropy_with_logits(input, target, weight)

    def forward(self, input, target):
        return self.balanced_bce_loss(input, target)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, input, target):
        prob = input.sigmoid()
        target = target.type_as(input)

        # focal weights
        pt = (1 - prob) * target + prob * (1 - target)
        weight = pt.pow(self.gamma) * (self.alpha * target + (1 - self.alpha) * (1 - target))

        # BCE loss with focal weights
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none') * weight

        return loss.mean()

    def forward(self, input, target):
        return self.focal_loss(input, target)


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def smooth_l1_loss(self, input, target):
        diff = torch.abs(input - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta)
        return loss.mean()

    def forward(self, input, target):
        return self.smooth_l1_loss(input, target)


