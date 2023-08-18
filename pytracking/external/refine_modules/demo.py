import os
import numpy as np
import cv2
import random

from pytracking.refine_modules.refine_module import RefineModule
from pytracking.refine_modules.utils import add_frame_bbox, add_frame_mask


COLORS = [
    (72.8571, 255.0000, 0),     # gt
    (255.0000, 218.5714, 0),  # base tracker
    (0, 145.7143, 255.0000),  # iou-net
    (0, 255.0000, 145.7143),    # siammask
    (255.0000, 0, 0),         # ar
    (72.8571, 0, 255.0000),
    (255.0000, 0, 218.5714),
]

def main():
    """ refinement module testing code """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    refine_path = '/media/hexdjx/907856427856276E/networks/SEcmnet_ep0040-c.pth.tar'
    SE_module = RefineModule(refine_path, 0)
    video_dir = '/media/hexdjx/907856427856276E/OTB100/Basketball'
    color = np.array(COLORS[random.randint(0, len(COLORS) - 1)])[None, None, ::-1]

    gt_file = os.path.join(video_dir, 'groundtruth_rect.txt')

    gt = np.loadtxt(gt_file, dtype=np.float32, delimiter=',')
    frame1_path = os.path.join(video_dir, 'img/0001.jpg')
    # frame1_path = os.path.join(video_dir, 'img','00000001.jpg')
    frame1 = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
    SE_module.initialize(frame1, gt[0])
    for i in range(1, gt.shape[0]):
        # print(i)
        # frame_path = os.path.join(video_dir,'img', '%08d.jpg'%(i+1))
        frame_path = os.path.join(video_dir, 'img/%04d.jpg' % (i + 1))

        frame = cv2.imread(frame_path)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_dict = SE_module.refine(frame_RGB, gt[i], test=True)  # default corner

        mask_pred = SE_module.get_mask(frame_RGB, gt[i])
        # from pytracking.pysot_toolkit.toolkit.visualization.draw_mask import draw_mask
        # draw_mask(frame, mask_pred, idx=i, show=True)


        im4show = frame
        mask_pred = np.uint8(mask_pred > 0.5)[:, :, None]
        contours, _ = cv2.findContours(mask_pred.squeeze(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im4show = im4show * (1 - mask_pred) + np.uint8(im4show * mask_pred / 2) + mask_pred * np.uint8(color) * 128

        cv2.drawContours(im4show, contours, -1, color[::-1].squeeze(), 2)
        cv2.putText(im4show, str(i), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('', im4show)
        cv2.waitKey(1)
        '''add bbox'''
        # frame = add_frame_bbox(frame, output_dict['bbox'], (255, 0, 0))
        '''add mask'''
        # frame = add_frame_mask(frame, output_dict['mask'], 0.5)
        '''add mask bbox'''
        # frame = add_frame_bbox(frame,output_dict['mask_bbox'],(0,0,255))
        '''add corner'''
        # frame = add_frame_bbox(frame, output_dict['corner'], (0, 255, 0))
        '''add fuse bbox'''
        # frame = add_frame_bbox(frame, output_dict['all'], (0, 0, 255))
        '''show'''
        # save_path = os.path.join(save_dir, 'img/%04d.jpg' % (i + 1))
        # cv2.imwrite(save_path, frame)
        # cv2.imshow('', frame)
        # cv2.waitKey(1)


if __name__ == '__main__':
    main()
