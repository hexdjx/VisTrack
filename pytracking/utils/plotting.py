import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from mpl_toolkits.mplot3d import Axes3D
from pytracking.features.preprocessing import torch_to_numpy

def draw_figure(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)


def show_tensor(a: torch.Tensor, fig_num=None, title=None, range=(None, None), ax=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))

    if ax is None:
        fig = plt.figure(fig_num)
        plt.tight_layout()
        plt.cla()
        plt.imshow(a_np, vmin=range[0], vmax=range[1])
        plt.axis('off')
        plt.axis('equal')
        if title is not None:
            plt.title(title)

        draw_figure(fig)
    else:
        ax.cla()
        ax.imshow(a_np, vmin=range[0], vmax=range[1])
        ax.set_axis_off()
        ax.axis('equal')
        if title is not None:
            ax.set_title(title)
        draw_figure(plt.gcf())


def plot_graph(a: torch.Tensor, fig_num=None, title=None):
    """Plot graph. Data is a 1D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim > 1:
        raise ValueError
    fig = plt.figure(fig_num)
    # plt.tight_layout()
    plt.cla()
    plt.plot(a_np)
    if title is not None:
        plt.title(title)
    draw_figure(fig)


def show_image_with_boxes(im, boxes, iou_pred=None, disp_ids=None):
    im_np = im.clone().cpu().squeeze().numpy()
    im_np = np.ascontiguousarray(im_np.transpose(1, 2, 0).astype(np.uint8))

    boxes = boxes.view(-1, 4).cpu().numpy().round().astype(int)

    # Draw proposals
    for i_ in range(boxes.shape[0]):
        if disp_ids is None or disp_ids[i_]:
            bb = boxes[i_, :]
            disp_color = (i_ * 38 % 256, (255 - i_ * 97) % 256, (123 + i_ * 66) % 256)
            cv2.rectangle(im_np, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                          disp_color, 1)

            if iou_pred is not None:
                text_pos = (bb[0], bb[1] - 5)
                cv2.putText(im_np, 'ID={} IOU = {:3.2f}'.format(i_, iou_pred[i_]), text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, bottomLeftOrigin=False)

    im_tensor = torch.from_numpy(im_np.transpose(2, 0, 1)).float()

    return im_tensor


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    """ Overlay mask over image.
    Source: https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/utils/visualization.py
    This function allows you to overlay a mask over an image with some
    transparency.
    # Arguments
        im: Numpy Array. Array with the image. The shape must be (H, W, 3) and
            the pixels must be represented as `np.uint8` data type.
        ann: Numpy Array. Array with the mask. The shape must be (H, W) and the
            values must be integers
        alpha: Float. Proportion of alpha to apply at the overlaid mask.
        colors: Numpy Array. Optional custom colormap. It must have shape (N, 3)
            being N the maximum number of colors to represent.
        contour_thickness: Integer. Thickness of each object index contour draw
            over the overlay. This function requires to have installed the
            package `opencv-python`.
    # Returns
        Numpy Array: Image of the overlay with shape (H, W, 3) and data type
            `np.uint8`.
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img

#########################################################################################
# my add
# visualize respond map in 3D
def mesh_score(s, fig_num=None, frame_num=None):
    score = s.squeeze().cpu().clone().detach().numpy()

    [x, y] = np.shape(score)
    fig = plt.figure(fig_num)
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
    # plt.show()
    plt.savefig('/home/hexd6/code/Tracking/VisTrack/pytracking/analysis/utils/results/fudimp/{:08d}.png'
                .format(frame_num), format='png', dpi=300)


def plot_scatter(psr, apce, psmd, det):
    x_frame = [i + 1 for i in range(len(psr))]
    plt.plot(x_frame, psr, color='red', linewidth=2.0, linestyle='-')
    plt.plot(x_frame, apce, color='green', linewidth=2.0, linestyle='--')
    plt.plot(x_frame, psmd, color='blue', linewidth=2.0, linestyle='-.')
    # plt.plot(x_frame, det, color='black', linewidth=2.0, linestyle=':')

    plt.title('Score index', fontsize=24)
    plt.xlabel('video frame')
    plt.ylabel('score index')

    # 保存图片到本地
    # plt.savefig('scale_precision.png', dpi=300, bbox_inches='tight')
    plt.show()

def im_show(im, name):
    im_patch = torch_to_numpy(im)

    plt.figure(10)
    plt.tight_layout()
    plt.cla()
    plt.imshow(im_patch.astype(np.int))
    plt.axis('off')
    plt.axis('equal')
    plt.show()
    # plt.savefig('/home/hexdjx/code/Tracking/pytracking/pytracking/results/img/{}.png'.format(name),
    #             format='png', dpi=300)

def im_save(im, name):
    im_patch = torch_to_numpy(im)
    from PIL import Image
    img = Image.fromarray(np.uint8(im_patch))
    img.save('/home/hexdjx/code/Tracking/pytracking/pytracking/results/img/{}.png'.format(name))
#########################################################################################
