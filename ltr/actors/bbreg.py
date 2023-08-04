import torch
from . import BaseActor


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats


class AtomBBKLActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM with BBKL"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_density', and 'gt_density'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        bb_scores = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        bb_scores = bb_scores.view(-1, bb_scores.shape[2])
        proposal_density = data['proposal_density'].view(-1, data['proposal_density'].shape[2])
        gt_density = data['gt_density'].view(-1, data['gt_density'].shape[2])

        # Compute loss
        loss = self.objective(bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': loss.item()}

        return loss, stats


# --Scale Estimator-- ############################################
"""bounding box regressor"""

class CornerActor(BaseActor):
    """ Actor for training the our scale estimation module"""
    def __init__(self, net, objective, loss_type=None):
        super().__init__(net, objective)

        self.loss_type = loss_type

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno', 'test_anno'

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain bbox prediction for each test image'
        pred_dict = self.net(data['train_images'], data['test_images'], data['train_anno'])
        corner_pred = pred_dict['corner']  # (batch,4)

        # get groundtruth
        bbox_gt = data['test_anno'].squeeze(0)  # (x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)

        # get loss function
        loss = self.objective[self.loss_type](corner_pred, bbox_gt_xyxy)  # smooth_l1/iou/giou/diou/ciou

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        stats = {
            'Loss/total_loss': loss.item()
        }

        return loss, stats


class MaskActor(BaseActor):
    """ Actor for training the our scale estimation module about mask"""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'mask': 1.0}
        self.loss_weight = loss_weight
        
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_anno', 'test_masks'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        pred_dict = self.net(data['train_images'], data['test_images'], data['train_anno'])
        mask_pred = pred_dict['mask']  # (batch,1,w,h)

        mask_gt = data['test_masks'].squeeze(0)
        # Compute loss for mask

        loss = self.objective['mask'](mask_pred, mask_gt) * self.loss_weight['mask']

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        stats = {
            'Loss/total': loss.item(),
            'Loss/mask': loss.item(),
        }

        return loss, stats


class CornerMaskActor(BaseActor):
    """ Actor for training the our scale estimation module"""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'smooth_l1': 1.0, 'giou': 1.0, 'mask': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_anno', 'test_masks'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain bbox prediction for each test image'
        '''get prediction'''
        pred_dict = self.net(data['train_images'], data['test_images'], data['train_anno'])
        corner_pred = pred_dict['corner']  # (batch,4)
        mask_pred = pred_dict['mask']  # (batch,1,256,256)

        '''get groundtruth'''
        bbox_gt = data['test_anno'].squeeze(0)  # 测试帧的真值框在裁剪出的搜索区域上的坐标(x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)格式

        # get corner loss function
        smooth_l1_loss = self.objective['smooth_l1'](corner_pred, bbox_gt_xyxy) * self.loss_weight['smooth_l1']
        giou_loss = self.objective['giou'](corner_pred, bbox_gt_xyxy) * self.loss_weight['giou']

        mask_flag = (data['mask'] == 1)  # data[mask]是一个tensor,有batch个元素,其中有的是0有的是1. mask_flag是一个bool型tensor
        num_mask_sample = mask_flag.sum()
        if num_mask_sample > 0:
            mask_gt = data['test_masks'].squeeze(0)  # 测试帧的mask真值框 (batch,1,H,W)
            '''Compute loss for mask'''
            loss_mask = self.loss_weight['mask'] * self.objective['mask'](mask_pred[mask_flag], mask_gt[mask_flag])  # 只计算那些mask_flag等于1的样本的loss_mask
        else:
            loss_mask = torch.zeros((1,)).cuda()

        loss = smooth_l1_loss + giou_loss + loss_mask

        stats = {
            'Loss/total': loss.item(),
            'Loss/smooth_l1_loss': smooth_l1_loss.item(),
            'Loss/giou_loss': giou_loss.item(),
            'Loss/mask': loss_mask.item(),
        }

        return loss, stats
####################################################################################

