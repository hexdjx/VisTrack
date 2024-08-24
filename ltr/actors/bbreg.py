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


# --corner head---------------------------------------------
class CornerActor(BaseActor):
    """ Actor for training the scale estimation module"""

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno', 'test_anno'

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain bbox prediction for each test image
        corner_pred = self.net(data['train_images'], data['test_images'], data['train_anno'])

        # get groundtruth
        bbox_gt = data['test_anno'].squeeze(0)  # (x1,y1,w,h)

        bbox_gt_xyxy = bbox_gt.clone()
        bbox_gt_xyxy[:, 2:] += bbox_gt_xyxy[:, :2]  # (x1,y1,x2,y2)

        giou_loss = self.objective['giou'](corner_pred, bbox_gt_xyxy)
        l1_loss = self.objective['l1'](corner_pred, bbox_gt_xyxy)
        loss = 2.0 * giou_loss + 5.0 * l1_loss

        stats = {
            'Loss/total': loss.item(),
            'Loss/l1_loss': l1_loss.item(),
            'Loss/giou_loss': giou_loss.item()
        }

        return loss, stats
# ----------------------------------------------------------


