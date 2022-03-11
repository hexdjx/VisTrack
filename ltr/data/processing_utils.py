import numpy as np
import torch
import math
import cv2 as cv
import random
import torch.nn.functional as F
from .bounding_box_utils import rect_to_rel, rel_to_rect


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    x, y, w, h = target_bb.tolist()

    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

        if mask is None:
            return im_crop_padded, resize_factor
        mask_crop_padded = \
            F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[
                0, 0]
        return im_crop_padded, resize_factor, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, 1.0
        return im_crop_padded, 1.0, mask_crop_padded


def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, masks_crop = zip(*crops_resize_factors)

    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]

    return frames_crop, box_crop, masks_crop


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        if w_per <= 1:
            w_per = box[2] * rand_uniform(0.15, 0.5)

        if h_per <= 1:
            h_per = box[3] * rand_uniform(0.15, 0.5)

        box_per = torch.Tensor([c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per]).round()

        if box_per[2] <= 1:
            box_per[2] = box[2] * rand_uniform(0.15, 0.5)

        if box_per[3] <= 1:
            box_per[3] = box[3] * rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou


def sample_target_adaptive(im, target_bb, search_area_factor, output_sz, mode: str = 'replicate',
                           max_scale_change=None, mask=None):
    """ Extracts a crop centered at target_bb box, of area search_area_factor^2. If the crop area contains regions
    outside the image, it is shifted so that the it is inside the image. Further, if the crop area exceeds the image
    size, a smaller crop which fits the image is returned instead.

    args:
        im - Input numpy image to crop.
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        mask - Optional mask to apply the same crop.

    returns:
        numpy image - Extracted crop.
        torch.Tensor - A bounding box denoting the cropped region in the image.
        numpy mask - Cropped mask returned only if mask is not None.
    """

    if max_scale_change is None:
        max_scale_change = float('inf')
    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)
    output_sz = torch.Tensor(output_sz)

    im_h = im.shape[0]
    im_w = im.shape[1]

    bbx, bby, bbw, bbh = target_bb.tolist()

    # Crop image
    crop_sz_x, crop_sz_y = (output_sz * (
            target_bb[2:].prod() / output_sz.prod()).sqrt() * search_area_factor).ceil().long().tolist()

    # Get new sample size if forced inside the image
    if mode == 'inside' or mode == 'inside_major':
        # Calculate rescaling factor if outside the image
        rescale_factor = [crop_sz_x / im_w, crop_sz_y / im_h]
        if mode == 'inside':
            rescale_factor = max(rescale_factor)
        elif mode == 'inside_major':
            rescale_factor = min(rescale_factor)
        rescale_factor = min(max(1, rescale_factor), max_scale_change)

        crop_sz_x = math.floor(crop_sz_x / rescale_factor)
        crop_sz_y = math.floor(crop_sz_y / rescale_factor)

    if crop_sz_x < 1 or crop_sz_y < 1:
        raise Exception('Too small bounding box.')

    x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
    x2 = x1 + crop_sz_x

    y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
    y2 = y1 + crop_sz_y

    # Move box inside image
    shift_x = max(0, -x1) + min(0, im_w - x2)
    x1 += shift_x
    x2 += shift_x

    shift_y = max(0, -y1) + min(0, im_h - y2)
    y1 += shift_y
    y2 += shift_y

    out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
    out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
    shift_x = (-x1 - out_x) * (out_x > 0)
    shift_y = (-y1 - out_y) * (out_y > 0)

    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Resize image
    im_out = cv.resize(im_crop_padded, tuple(output_sz.long().tolist()))

    if mask is not None:
        mask_out = \
            F.interpolate(mask_crop_padded[None, None], tuple(output_sz.flip(0).long().tolist()), mode='nearest')[0, 0]

    crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    if mask is None:
        return im_out, crop_box
    else:
        return im_out, crop_box, mask_out


def crop_and_resize(im, box, crop_bb, output_sz, mask=None):
    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)

    im_h = im.shape[0]
    im_w = im.shape[1]

    if crop_bb[2] < 1 or crop_bb[3] < 1:
        raise Exception('Too small bounding box.')

    x1 = crop_bb[0]
    x2 = crop_bb[0] + crop_bb[2]

    y1 = crop_bb[1]
    y2 = crop_bb[1] + crop_bb[3]

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    # Resize image
    im_out = cv.resize(im_crop_padded, output_sz)

    if mask is not None:
        mask_out = F.interpolate(mask_crop_padded[None, None], (output_sz[1], output_sz[0]), mode='nearest')[0, 0]

    rescale_factor = output_sz[0] / crop_bb[2]

    # Hack
    if box is not None:
        box_crop = box.clone()
        box_crop[0] -= crop_bb[0]
        box_crop[1] -= crop_bb[1]

        box_crop *= rescale_factor
    else:
        box_crop = None

    if mask is None:
        return im_out, box_crop
    else:
        return im_out, box_crop, mask_out


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    return box_out


def target_image_crop(frames, box_extract, box_gt, search_area_factor, output_sz, mode: str = 'replicate',
                      max_scale_change=None, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. If the crop area contains regions outside the image, it is shifted / shrunk so that it
    completely fits inside the image. The extracted crops are then resized to output_sz. Further, the co-ordinates of
    the box box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        masks - Optional masks to apply the same crop.

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if isinstance(output_sz, (float, int)):
        output_sz = (output_sz, output_sz)

    if masks is None:
        frame_crops_boxes = [sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change)
                             for f, a in zip(frames, box_extract)]

        frames_crop, crop_boxes = zip(*frame_crops_boxes)
    else:
        frame_crops_boxes_masks = [
            sample_target_adaptive(f, a, search_area_factor, output_sz, mode, max_scale_change, mask=m)
            for f, a, m in zip(frames, box_extract, masks)]

        frames_crop, crop_boxes, masks_crop = zip(*frame_crops_boxes_masks)

    crop_sz = torch.Tensor(output_sz)

    # find the bb location in the crop
    box_crop = [transform_box_to_crop(bb_gt, crop_bb, crop_sz)
                for bb_gt, crop_bb in zip(box_gt, crop_boxes)]

    if masks is None:
        return frames_crop, box_crop
    else:
        return frames_crop, box_crop, masks_crop


def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:, :2], proposals[:, :2])
    br = torch.min(reference[:, :2] + reference[:, 2:], proposals[:, :2] + proposals[:, 2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True, density=False,
                            uni_bias=0):
    """Construct Gaussian label function."""

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = target_bb[:, 0:2] + 0.5 * target_bb[:, 2:4]
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label


def gauss_density_centered(x, std):
    """Evaluate the probability density of a Gaussian centered at zero.
    args:
        x - Samples.
        std - List of standard deviations
    """
    return torch.exp(-0.5 * (x / std) ** 2) / (math.sqrt(2 * math.pi) * std)


def gmm_density_centered(x, std):
    """Evaluate the probability density of a GMM centered at zero.
    args:
        x - Samples. Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
        std - Tensor of standard deviations
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)


def sample_gmm_centered(std, num_samples=1):
    """Sample from a GMM distribution centered at zero:
    args:
        std - Tensor of standard deviations
        num_samples - number of samples
    """
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0, :, k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    return x_centered, prob_dens


def sample_gmm(mean, std, num_samples=1):
    """Sample from a GMM distribution:
    args:
        mean - a single mean vector
        std - Tensor of standard deviations
        num_samples - number of samples
    """
    num_dims = mean.numel()
    num_components = std.shape[-1]

    mean = mean.view(1, num_dims)
    std = std.view(1, -1, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0, :, k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    x = x_centered + mean
    prob_dens = gmm_density_centered(x_centered, std)

    return x, prob_dens


def sample_box_gmm(mean_box, proposal_sigma, gt_sigma=None, num_samples=1, add_mean_box=False):
    """Sample boxes from a Gaussian mixture model.
    args:
        mean_box - Center (or mean) bounding box
        proposal_sigma - List of standard deviations for each Gaussian
        gt_sigma - Standard deviation of the ground truth distribution
        num_samples - Number of sampled boxes
        add_mean_box - Also add mean box as first element

    returns:
        proposals, proposal density and ground truth density for all samples
    """
    center_std = torch.Tensor([s[0] for s in proposal_sigma])
    sz_std = torch.Tensor([s[1] for s in proposal_sigma])
    std = torch.stack([center_std, center_std, sz_std, sz_std])

    mean_box = mean_box.view(1, 4)
    sz_norm = mean_box[:, 2:].clone()

    # Sample boxes
    proposals_rel_centered, proposal_density = sample_gmm_centered(std, num_samples)

    # Add mean and map back
    mean_box_rel = rect_to_rel(mean_box, sz_norm)
    proposals_rel = proposals_rel_centered + mean_box_rel
    proposals = rel_to_rect(proposals_rel, sz_norm)

    if gt_sigma is None or gt_sigma[0] == 0 and gt_sigma[1] == 0:
        gt_density = torch.zeros_like(proposal_density)
    else:
        std_gt = torch.Tensor([gt_sigma[0], gt_sigma[0], gt_sigma[1], gt_sigma[1]]).view(1, 4)
        gt_density = gauss_density_centered(proposals_rel_centered, std_gt).prod(-1)

    if add_mean_box:
        proposals = torch.cat((mean_box, proposals))
        proposal_density = torch.cat((torch.Tensor([-1]), proposal_density))
        gt_density = torch.cat((torch.Tensor([1]), gt_density))

    return proposals, proposal_density, gt_density

def find_local_maxima(scores, th, ks):
    """Find local maxima in a heat map.
        args:
            scores - heat map to find the local maxima in.
            th - threshold that defines the minamal value needed to be considered as a local maximum.
            ks = local neighbourhood (kernel size) specifiying the minimal distance between two maxima.

        returns:
            coordinates and values of the local maxima.
    """
    ndims = scores.ndim

    if ndims == 2:
        scores = scores.view(1, 1, scores.shape[0], scores.shape[1])

    scores_max = F.max_pool2d(scores, kernel_size=ks, stride=1, padding=ks//2)

    peak_mask = (scores == scores_max) & (scores > th)
    coords = torch.nonzero(peak_mask)
    intensities = scores[peak_mask]

    # Highest peak first
    idx_maxsort = torch.argsort(-intensities)
    coords = coords[idx_maxsort]
    intensities = intensities[idx_maxsort]

    if ndims == 4:

        coords_batch, intensities_batch, = TensorList(), TensorList()
        for i in range(scores.shape[0]):
            mask = (coords[:, 0] == i)
            coords_batch.append(coords[mask, 2:])
            intensities_batch.append(intensities[mask])
    else:
        coords_batch = coords[:, 2:]
        intensities_batch = intensities

    return coords_batch, intensities_batch

# --RVT-- #####################################################################
# a simple negative sampling method
def target_proposals(box):
    radius = torch.ceil(torch.sqrt(box[2] * box[3]) / 2)

    box_1 = torch.Tensor([box[0] - radius, box[1] - radius, box[2], box[3]]).round()
    box_2 = torch.Tensor([box[0] + radius, box[1] - radius, box[2], box[3]]).round()
    box_3 = torch.Tensor([box[0] + radius, box[1] + radius, box[2], box[3]]).round()
    box_4 = torch.Tensor([box[0] - radius, box[1] + radius, box[2], box[3]]).round()

    bboxs = torch.stack((box, box_1, box_2, box_3, box_4), dim=0)
    return bboxs


def target_proposals_gauss(box, max_iou=0.3, sigma_factor=0.5):

    if not isinstance(sigma_factor, torch.Tensor):
        sigma_factor = sigma_factor * torch.ones(2)

    perturb_factor = torch.sqrt(box[2] * box[3]) * sigma_factor

    boxs = [box]
    # multiple tries to ensure that the perturbed box has iou < min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5 * box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        box_per = torch.Tensor([c_x_per - 0.5 * box[2], c_y_per - 0.5 * box[3], box[2], box[3]]).round()

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # reduce the perturb factor
        perturb_factor *= 1.1

        if 0 < box_iou < max_iou:
            boxs.append(box_per)
            if len(boxs) == 5:
                bboxs = torch.stack(boxs, dim=0)
                return bboxs

    # in case negative sample is insufficient
    radius = torch.ceil(torch.sqrt(box[2] * box[3]) / 2)
    box_1 = torch.Tensor([box[0] - radius, box[1] - radius, box[2], box[3]]).round()
    box_2 = torch.Tensor([box[0] + radius, box[1] - radius, box[2], box[3]]).round()
    box_3 = torch.Tensor([box[0] + radius, box[1] + radius, box[2], box[3]]).round()
    box_4 = torch.Tensor([box[0] - radius, box[1] + radius, box[2], box[3]]).round()
    boxs.append(box_1)
    boxs.append(box_2)
    boxs.append(box_3)
    boxs.append(box_4)
    boxs = boxs[:5]
    bboxs = torch.stack(boxs, dim=0)

    return bboxs


def crop_target_proposals(im, anno, output_sz=128):
    x, y, w, h = anno.tolist()

    x1 = round(x)
    x2 = round(x1 + w)

    y1 = round(y)
    y2 = round(y1 + h)

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

    im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))

    return im_crop_padded


# --AlphaRefine--#####################################################################
#  for scale estimation(no square)
def sample_target_v2(im, target_bb, search_area_factor=1.0, output_sz=None, mode=cv.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, especially, search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """

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

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape) == 2:
            im_crop_padded_rsz = im_crop_padded_rsz[..., np.newaxis]
        return im_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, 1.0, 1.0


def transform_image_to_crop_v2(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor_h: float,
                               resize_factor_w: float, crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_xc = (crop_sz[0] - 1) / 2 + (box_in_center[0] - box_extract_center[0]) * resize_factor_w
    box_out_yc = (crop_sz[1] - 1) / 2 + (box_in_center[1] - box_extract_center[1]) * resize_factor_h
    box_out_w = box_in[2] * resize_factor_w
    box_out_h = box_in[3] * resize_factor_h

    '''2019.12.28 为了避免出现(x1,y1)小于0,或者(x2,y2)大于256的情况,这里我对它们加上了一些限制
    max_sz = crop_sz[0].item()
    box_out_x1 = torch.clamp(box_out_xc - 0.5 * box_out_w,0,max_sz)
    box_out_y1 = torch.clamp(box_out_yc - 0.5 * box_out_h,0,max_sz)
    box_out_x2 = torch.clamp(box_out_xc + 0.5 * box_out_w,0,max_sz)
    box_out_y2 = torch.clamp(box_out_yc + 0.5 * box_out_h,0,max_sz)
    box_out_w_new = box_out_x2 - box_out_x1
    box_out_h_new = box_out_y2 - box_out_y1
    box_out = torch.stack((box_out_x1, box_out_y1, box_out_w_new, box_out_h_new))
    '''
    box_out = torch.cat((box_out_xc - 0.5 * box_out_w, box_out_yc - 0.5 * box_out_h, box_out_w, box_out_h))
    return box_out


def jittered_center_crop_v2(frames, box_extract, box_gt, search_area_factor, output_sz, get_bbox_coord=True,
                            mode=cv.BORDER_REPLICATE):
    """
    Crop a patch centered at box_extract. The height and width of cropped region is search_area_factor times that of box_extract.
    The extracted crops are then resized to output_sz. Further, the co-ordinates of the box box_gt are transformed to the image crop co-ordinates
    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
    """

    crops_resize_factors = [sample_target_v2(f, a, search_area_factor, output_sz, mode=mode)
                            for f, a in zip(frames, box_extract)]

    frames_crop, resize_factors_h, resize_factors_w = zip(*crops_resize_factors)
    if get_bbox_coord:
        crop_sz = torch.Tensor([output_sz, output_sz])

        # find the bb location in the crop
        '''get GT's cooridinate on the cropped patch'''
        box_crop = [transform_image_to_crop_v2(a_gt, a_ex, h_rsf, w_rsf, crop_sz)
                    for a_gt, a_ex, h_rsf, w_rsf in zip(box_gt, box_extract, resize_factors_h, resize_factors_w)]

        return frames_crop, box_crop
    else:
        return frames_crop

#################################################################################

# --MDNet--#################################################
# function: sampling required pos/neg samples
def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


# for example:
# pos_examples = SampleGenerator('gaussian', image.size, 0.1, 1.3)(
#             target_bbox, 500, [0.7, 1])

# pos_examples = SampleGenerator('gaussian', image.size, 0.1, 1.3)(
#             target_bbox, 500, [0.7, 1])
class SampleGenerator():
    def __init__(self, type_, img_size, trans=1, scale=1, aspect=None, valid=False):
        self.type = type_
        self.img_size = np.array(img_size)  # (w, h)
        self.trans = trans
        self.scale = scale
        self.aspect = aspect
        self.valid = valid

    def _gen_samples(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))

        # vary aspect ratio
        if self.aspect is not None:
            ratio = np.random.rand(n, 2) * 2 - 1
            samples[:, 2:] *= self.aspect ** ratio

        # sample generation
        if self.type == 'gaussian':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * np.clip(0.5 * np.random.randn(n, 2), -1, 1)
            samples[:, 2:] *= self.scale ** np.clip(0.5 * np.random.randn(n, 1), -1, 1)

        elif self.type == 'uniform':
            samples[:, :2] += self.trans * np.mean(bb[2:]) * (np.random.rand(n, 2) * 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        elif self.type == 'whole':
            m = int(2 * np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))).reshape(-1, 2)
            xy = np.random.permutation(xy)[:n]
            samples[:, :2] = bb[2:] / 2 + xy * (self.img_size - bb[2:] / 2 - 1)
            samples[:, 2:] *= self.scale ** (np.random.rand(n, 1) * 2 - 1)

        # adjust bbox range
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        if self.valid:
            samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2, self.img_size - samples[:, 2:] / 2 - 1)
        else:
            samples[:, :2] = np.clip(samples[:, :2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:, :2] -= samples[:, 2:] / 2

        return samples

    def __call__(self, bbox, n, overlap_range=None, scale_range=None):

        if overlap_range is None and scale_range is None:
            return self._gen_samples(bbox, n)

        else:
            samples = None
            remain = n
            factor = 2
            while remain > 0 and factor < 16:
                samples_ = self._gen_samples(bbox, remain * factor)

                idx = np.ones(len(samples_), dtype=bool)
                if overlap_range is not None:
                    r = overlap_ratio(samples_, bbox)
                    idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
                if scale_range is not None:
                    s = np.prod(samples_[:, 2:], axis=1) / np.prod(bbox[2:])
                    idx *= (s >= scale_range[0]) * (s <= scale_range[1])

                samples_ = samples_[idx, :]
                samples_ = samples_[:min(remain, len(samples_))]
                if samples is None:
                    samples = samples_
                else:
                    samples = np.concatenate([samples, samples_])
                remain = n - len(samples)
                factor = factor * 2

            return samples

    def set_type(self, type_):
        self.type = type_

    def set_trans(self, trans):
        self.trans = trans

    def expand_trans(self, trans_limit):
        self.trans = min(self.trans * 1.1, trans_limit)
############################################################
