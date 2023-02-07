import math
import torch
import torch.nn as nn


# ToMP
class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=None):
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)

        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 4)  # nf x ns x x 4 x h x w
        target = target.permute(0, 1, 3, 4, 2).reshape(-1, 4)  # nf x ns x 4 x h x w

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-7
        ious = (area_intersect) / (area_union)
        gious = ious - (ac_union - area_union) / ac_union

        losses = 1 - gious

        if weights is not None and weights.sum() > 0:
            weights = weights.permute(0, 1, 3, 4, 2).reshape(-1)  # nf x ns x x 1 x h x w
            loss_mean = losses[weights > 0].mean()
            ious = ious[weights > 0]
        else:
            loss_mean = losses.mean()

        return loss_mean, ious


################################################
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1]  # (N,)

    return iou - (area - union) / area


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean()


# 输入格式 (l,t,r,b)
class IoULoss(nn.Module):
    def __init__(self, loc_loss_type='iou'):
        super(IoULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def iou_loss(self, input, target, weight=None):
        # assume the order of [x1, y1, x2, y2]
        il, ir, it, ib = input.t()
        tl, tr, tt, tb = target.t()

        input_area = (il + ir) * (it + ib)
        target_area = (tl + tr) * (tt + tb)

        inter_w = torch.min(il, tl) + torch.min(ir, tr)
        inter_h = torch.min(ib, tb) + torch.min(it, tt)

        # giou
        inter_w_g = torch.max(il, tl) + torch.max(ir, tr)
        inter_h_g = torch.max(ib, tb) + torch.max(it, tt)
        enclose_area = inter_w_g * inter_h_g

        inter_area = inter_w * inter_h
        union_area = input_area + target_area - inter_area

        iou = (inter_area + 1.0) / (union_area + 1.0)

        giou = iou - (enclose_area - union_area) / enclose_area

        if self.loc_loss_type == 'iou':
            loss = -torch.log(iou)
        elif self.loc_loss_type == 'linear_iou':
            loss = 1 - iou
        elif self.loc_loss_type == 'giou':
            loss = 1 - giou
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()
        else:
            return loss.mean()

    def forward(self, input, target, weight=None):
        return self.iou_loss(input, target, weight)


# 输入格式 (x1,y1,x2,y2)
class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        """form: (x1,y1,x2,y2)"""
        pred_x1 = pred[:, 0]
        pred_y1 = pred[:, 1]
        pred_x2 = pred[:, 2]
        pred_y2 = pred[:, 3]

        target_x1 = target[:, 0]
        target_y1 = target[:, 1]
        target_x2 = target[:, 2]
        target_y2 = target[:, 3]

        '''compute DIOU'''
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (pred[:, :2] + pred[:, 2:]) / 2
        d = torch.sum(torch.square(pred_center - target_center))

        '''Compute seperate areas'''
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        '''Compute intersection area and iou'''
        w_inter = torch.clamp(torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1), 0)
        h_inter = torch.clamp(torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1), 0)
        area_intersect = w_inter * h_inter
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        '''Compute C and Giou'''
        W_c = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        H_c = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        ac_uion = W_c * H_c + 1e-7

        giou_term = (ac_uion - area_union) / ac_uion
        gious = ious - giou_term

        diou_term = d / (W_c ** 2 + H_c ** 2)
        dious = ious - diou_term

        v = 4 / math.pi ** 2 * torch.pow(torch.atan((target_x2 - target_x1) / (target_y2 - target_y1)) - \
                                         torch.atan((pred_x2 - pred_x1) / (pred_y2 - pred_y1)), 2)
        alpha = v / (1 - ious + v)
        diou_term = diou_term + alpha * v
        cious = ious - diou_term

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        elif self.loss_type == 'diou':
            losses = 1 - dious
        elif self.loss_type == 'ciou':
            losses = 1 - cious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            '''use mean rather than sum'''
            return losses.mean()
