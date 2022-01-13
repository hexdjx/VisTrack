import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# rename file name
base_path = '/media/hexdjx/907856427856276E/NFS/anno/'
def file_rename(path):
    file_list = os.listdir(base_path)
    for f in file_list:
        old_path = os.path.join(base_path, f)
        # new_name = f.split('.')
        # new_name = 'nfs_' + new_name[0]
        new_name = f + '.txt'
        new_path = os.path.join(base_path, new_name)
        os.rename(old_path, new_path)
# file_rename(base_path)


# The posterior probability of target, please refer to staple tracker
img_path = '/media/hexdjx/907856427856276E/OTB100/Basketball/img/0001.jpg'
bbox = np.array([198, 214, 34, 81])  #198, 214, 34, 81 /187, 233, 34, 81
img = np.array(Image.open(img_path))

inner_padding = 0.2
# learning_rate_pwp = 0.04
pos = np.array([bbox[1] + (bbox[3] - 1) / 2, bbox[0] + (bbox[2] - 1) / 2])
target_sz = bbox[2:][::-1]
avg_dim = target_sz.sum() / 2
bg_area = np.floor(target_sz + avg_dim)
fg_area = np.floor(target_sz - avg_dim * inner_padding)
bg_area[0] = np.minimum(bg_area[0], img.shape[0] - 1)
bg_area[1] = np.minimum(bg_area[1], img.shape[1] - 1)

bg_area = bg_area - (bg_area - target_sz) % 2
fg_area = fg_area + (bg_area - fg_area) % 2


xs = np.floor(pos[1] + [np.array([0, bg_area[1]]) - bg_area[1] / 2])
ys = np.floor(pos[0] + [np.array([0, bg_area[0]]) - bg_area[0] / 2])

xs[xs < 1] = 1
ys[ys < 1] = 1
xs[xs > np.size(img, 1)] = np.size(img, 1)
ys[ys > np.size(img, 0)] = np.size(img, 0)
xs = xs.reshape(-1).astype(np.int16)
ys = ys.reshape(-1).astype(np.int16)

im_patch = img[ys[0]:ys[1], xs[0]:xs[1], :]
# plt.imshow(im_patch)

pad_offset1 = (bg_area - target_sz) / 2
pad_offset1 = pad_offset1.astype(np.int16)
bg_mask = np.ones(bg_area.astype(np.int16))
pad_offset1[pad_offset1 <= 0] = 1
bg_mask[pad_offset1[0]:-pad_offset1[0] + 1, pad_offset1[1]:-pad_offset1[1] + 1] = 0

pad_offset2 = (bg_area - fg_area) / 2
pad_offset2 = pad_offset2.astype(np.int16)
fg_mask = np.zeros_like(bg_mask)
pad_offset2[pad_offset2 <= 0] = 1
fg_mask[pad_offset2[0]:-pad_offset2[0] + 1, pad_offset2[1]:-pad_offset2[1] + 1] = 1

def computeHistogram(im_patch, mask, n_bins=16):
    [h, w, d] = im_patch.shape
    assert np.all((h, w) == mask.shape), 'mask and image are not the same size'
    bin_width = 256 / n_bins
    patch_array = np.reshape(im_patch, [w * h, d])
    bin_indices = np.floor(patch_array / bin_width) + 1
    counts = []
    mask = mask.reshape(-1, 1)
    mask_r = mask.repeat(3).reshape(-1, 3)
    bin_indices_m = bin_indices * mask_r
    for i in range(1, n_bins + 1):
        c = sum(i == bin_indices_m)
        counts.append(c)
    counts = np.stack(counts)
    counts_p = counts / sum(mask)
    img_p = np.zeros_like(bin_indices_m)
    c_s = bin_indices.shape
    for i in range(c_s[0]):
        for j in range(c_s[1]):
            col = bin_indices[i, j] - 1
            img_p[i, j] = counts_p[int(col), j]
    img_p = np.reshape(img_p, [h, w, d])
    img_p = img_p.mean(axis=-1)
    return img_p


# mask = np.ones(bg_area.astype(np.int16))
# p = computeHistogram(im_patch, mask)
fg_p = computeHistogram(im_patch, fg_mask)
bg_p = computeHistogram(im_patch, bg_mask)
P_O = fg_p / (fg_p + bg_p)
# P_O[P_O>=0.5]=1
# P_O[P_O<0.5]=0
plt.imshow(P_O)
plt.show()

