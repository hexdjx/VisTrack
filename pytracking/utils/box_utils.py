"""
Utilities for bounding box manipulation.
"""
import torch
import numpy as np
import cv2
import math


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def convert_vot_anno_to_rect(vot_anno, type='preserve_area'):
    if len(vot_anno) == 4:
        return vot_anno

    if type == 'union':
        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])
        return [x1, y1, x2 - x1, y2 - y1]
    elif type == 'preserve_area':
        if len(vot_anno) != 8:
            raise ValueError

        vot_anno = np.array(vot_anno)
        cx = np.mean(vot_anno[0::2])
        cy = np.mean(vot_anno[1::2])

        x1 = min(vot_anno[0::2])
        x2 = max(vot_anno[0::2])
        y1 = min(vot_anno[1::2])
        y2 = max(vot_anno[1::2])

        A1 = np.linalg.norm(vot_anno[0:2] - vot_anno[2: 4]) * np.linalg.norm(vot_anno[2: 4] - vot_anno[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1

        x = cx - 0.5 * w
        y = cy - 0.5 * h
        return [x, y, w, h]
    else:
        raise ValueError


def box_cxcywh_to_xyxy(x):  # torch.Tensor
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
    """boundary img.shape (H,W)"""
    x1_new = max(0, min(x1, boundary[1] - min_sz))
    y1_new = max(0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    return x1_new, y1_new, x2_new, y2_new


def mask_from_rect(rect, output_sz):
    """
    create a binary mask from a given rectangle
    rect: axis-aligned rectangle [x0, y0, width, height]
    output_sz: size of the output [width, height]
    """
    mask = np.zeros((output_sz[1], output_sz[0]), dtype=np.uint8)
    x0 = max(int(round(rect[0])), 0)
    y0 = max(int(round(rect[1])), 0)
    x1 = min(int(round(rect[0] + rect[2])), output_sz[0])
    y1 = min(int(round(rect[1] + rect[3])), output_sz[1])
    mask[y0:y1, x0:x1] = 1
    return mask


def rect_from_mask(mask):
    """
    create an axis-aligned rectangle from a given binary mask
    in created as a minimal rectangle containing all non-zero pixels
    """
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((1, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def mask2bbox(mask, ori_bbox, MASK_THRESHOLD=0.5):
    target_mask = (mask > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boundingRect(polygon)
    else:  # empty mask
        prbox = ori_bbox
    return np.array(prbox).astype(np.float)


def add_frame_mask(frame, mask, threshold=0.5):
    mask_new = (mask > threshold) * 255  # (H,W)
    frame_new = frame.copy().astype(np.float)
    frame_new[..., 1] += 0.3 * mask_new
    frame_new = frame_new.clip(0, 255).astype(np.uint8)
    return frame_new


def add_frame_bbox(frame, refined_box, color):
    x1, y1, w, h = refined_box.tolist()
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    return frame


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh


'''把mask映射到原图上'''
def map_mask_back(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    H, W = (im.shape[0], im.shape[1])
    base = np.zeros((H, W))
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - ws * 0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    '''pad base'''
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    '''Resize mask'''
    mask_rsz = cv2.resize(mask, (ws, hs))
    '''fill region with mask'''
    base_padded[y1 + y1_pad:y2 + y1_pad, x1 + x1_pad:x2 + x1_pad] = mask_rsz.copy()
    '''crop base_padded to get final mask'''
    final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
    assert (final_mask.shape == (H, W))
    return final_mask
