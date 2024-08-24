import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ltr.data.processing_utils as prutils
from pytracking import dcf


def get_target_probability(img, target_sz, inner_padding=0.2):
    def calHist_mask(im, mask, n_bins=16):
        h, w, c = im.shape

        hist_mask = [
            torch.Tensor(cv.calcHist([im.numpy()], [i], mask.numpy().astype(np.uint8), [16], [0, 256])) for i
            in range(c)]

        counts = torch.stack(hist_mask).squeeze().T

        counts_p = counts / mask.sum()

        bin_width = 256 / n_bins
        bin_indices = torch.floor(im.reshape(-1, c) / bin_width)

        for i in range(n_bins):
            for j in range(c):
                bin_indices[:, j][i == bin_indices[:, j]] = counts_p[i, j]

        img_p = bin_indices.reshape(h, w, c)
        # img_p = img_p.mean(axis=-1)

        return img_p

    img = img.permute(1, 2, 0)

    h, w, c = img.shape

    avg_dim = 0.5 * target_sz.sum()
    fg_area = torch.round(target_sz - avg_dim * inner_padding)

    bg_area = torch.Tensor([h, w])
    fg_area = fg_area + (bg_area - fg_area) % 2

    bg_mask = torch.ones([h, w])

    pad_offset1 = (bg_area - target_sz) / 2
    pad_offset1 = pad_offset1.int()
    pad_offset1[pad_offset1 <= 0] = 1
    bg_mask[pad_offset1[0]:-pad_offset1[0] + 1, pad_offset1[1]:-pad_offset1[1] + 1] = 0

    fg_mask = torch.zeros([h, w])

    pad_offset2 = (bg_area - fg_area) / 2
    pad_offset2 = pad_offset2.int()
    pad_offset2[pad_offset2 <= 0] = 1
    fg_mask[pad_offset2[0]:-pad_offset2[0] + 1, pad_offset2[1]:-pad_offset2[1] + 1] = 1

    fg_p = calHist_mask(img, fg_mask)
    bg_p = calHist_mask(img, bg_mask)

    P = fg_p / (fg_p + bg_p)

    P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)
    P = torch.where(torch.isinf(P), torch.full_like(P, 0), P)

    # plt.figure(10)
    # plt.subplot(311)
    # plt.imshow(fg_p)
    # plt.subplot(312)
    # plt.imshow(bg_p)
    # plt.set_cmap('binary')
    # plt.imshow(fg_mask)
    # plt.subplot(313)
    # plt.subplot(224)
    # plt.imshow(bg_mask)
    # plt.figure(10)
    # plt.imshow(img.int())
    # plt.imshow(torch.mean(P, dim=-1, keepdim=True))
    # plt.imshow(P[:,:,2])
    # plt.axis('off')
    # plt.colorbar()
    # plt.show()

    # return P.unsqueeze(-1)

    return P.permute(2, 0, 1).unsqueeze(0)


# for test use
def get_target_prob(img, bbox, inner_padding=0.0):
    def calHist_mask(im, mask, n_bins=16):

        h, w, c = im.shape

        hist_mask = [
            torch.Tensor(cv.calcHist([im], [i], mask.numpy().astype(np.uint8), [16], [0, 256])) for i
            in range(c)]

        counts = torch.stack(hist_mask).squeeze().T

        counts_p = counts / mask.sum()

        bin_width = 256 / n_bins
        bin_indices = torch.floor(torch.from_numpy(im).reshape(-1, c) / bin_width)

        for i in range(n_bins):
            for j in range(c):
                bin_indices[:, j][i == bin_indices[:, j]] = counts_p[i, j]

        img_p = bin_indices.reshape(h, w, c)

        return img_p

    h, w, c = img.shape

    x, y, bw, bh = bbox.tolist()

    bg_mask = torch.ones([h, w], dtype=torch.bool)
    bg_mask[round(y):round(y + bh), round(x):round(x + bw)] = False

    fg_mask = ~bg_mask

    # P = fg_p / (fg_p + bg_p + 0.01)

    # # Get position and size
    # pos = torch.Tensor([y + 0.5 * bh, x + 0.5 * bw])
    # target_sz = torch.Tensor([bh, bw])
    #
    # avg_dim = 0.5 * target_sz.sum()
    # fg_area = torch.round(target_sz - avg_dim * inner_padding)
    #
    # bg_area = torch.Tensor([h, w])
    # fg_area = fg_area + (bg_area - fg_area) % 2
    #
    # bg_mask = torch.ones([h, w])
    # bg_mask[round(y):round(y + bh), round(x):round(x + bw)] = 0
    #
    # y1x1 = torch.round(pos - 0.5 * fg_area)
    # y2x2 = torch.round(pos + 0.5 * fg_area)
    # y1, x1 = y1x1.int().tolist()
    # y2, x2 = y2x2.int().tolist()
    #
    # fg_mask = torch.zeros([h, w])
    # fg_mask[y1:y2, x1:x2] = 1

    # plt.figure()
    # plt.imshow(bg_mask.numpy())
    # plt.show()

    fg_p = calHist_mask(img, fg_mask)
    bg_p = calHist_mask(img, bg_mask)

    P = fg_p / (fg_p + bg_p + 0.01)

    output_window = dcf.hann2d(torch.tensor([288, 288]).long(), centered=True)

    P = P * output_window.squeeze().unsqueeze(-1)
    #
    P = cv.resize(P[..., 2].numpy(), (22, 22))
    # mesh_score(P)

    plt.figure()
    plt.imshow(P)
    plt.show()

    # P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)
    # P = torch.where(torch.isinf(P), torch.full_like(P, 0), P)

    return P.permute(2, 0, 1)


def bbox2mask(bbox):
    mask = torch.zeros([288, 288])
    x, y, w, h = bbox.tolist()

    x1 = round(x)
    y1 = round(y)
    x2 = round(x + w)
    y2 = round(y + h)
    # Crop target
    mask[y1:y2, x1:x2] = 1.0

    # plt.figure()
    # plt.imshow(mask.numpy())
    # plt.show()
    mesh_score(mask.numpy())

    return mask.unsqueeze(0)


def mesh_score(score):
    from mpl_toolkits.mplot3d import Axes3D
    [x, y] = np.shape(score)
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(0, x)
    Y = np.arange(0, y)
    X, Y = np.meshgrid(X, Y)
    Z = score
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', linewidth=0,
                           antialiased=False)  # rainbow coolwarm
    # ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=plt.get_cmap('rainbow'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_zlim(-0.2, 1.2)
    # ax.set_zlabel('Score')
    plt.axis('off')
    plt.show()


def get_iounet_box(sz):
    """All inputs in original image coordinates.
    Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
    box_center = (288 - 1) / 2
    box_sz = sz
    target_ul = box_center - (box_sz - 1) / 2
    return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


if __name__ == '__main__':
    fname = 'D:/Tracking/Datasets/OTB100/Basketball/img/0001.jpg'
    img = cv.cvtColor(cv.imread(fname), cv.COLOR_BGR2RGB)
    gt_state = torch.Tensor([198, 214, 34, 81])
    crops, boxes = prutils.target_image_crop([img], [gt_state], [gt_state], 5, 288)

    # im_crop = crops[0]
    # pred_s = boxes[0]
    # tl = tuple(map(int, [pred_s[0], pred_s[1]]))
    # br = tuple(map(int, [pred_s[0] + pred_s[2], pred_s[1] + pred_s[3]]))
    #
    # cv.rectangle(im_crop, tl, br, (0, 255, 0), 2)
    #
    # plt.figure()
    # plt.imshow(im_crop)
    # plt.show()

    # output_window = dcf.hann2d(torch.tensor([288, 288]).long(), centered=True)
    # plt.figure()
    # plt.imshow(output_window.squeeze().numpy())
    # plt.show()

    # mesh_score(output_window.squeeze().numpy())

    get_target_prob(crops[0], boxes[0])
    bbox2mask(boxes[0])

    target_box = get_iounet_box(gt_state[2:])
    target_label = prutils.gaussian_label_function(target_box.view(-1, 4),
                                                   1 / 4,
                                                   1,
                                                   18, 18 * 16)

    plt.figure()
    plt.imshow(target_label.permute(1, 2, 0).numpy())
    plt.show()
    mesh_score(target_label.squeeze().numpy())
