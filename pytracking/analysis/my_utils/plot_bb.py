import cv2 as cv
import os
import numpy as np
from pytracking.evaluation import get_dataset
import matplotlib.pyplot as plt

# _tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
#                         4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
#                         7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}
# _tracker_disp_colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255),
#                         4: (255, 255, 255), 5: (0, 0, 0), 6: (255, 128, 0)
#                         }

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 0, 0),
                        4: (0, 0, 255)
                        }

# default = 'tracking_results/ATOM/default/'
# multiscale = 'tracking_results/ATOM/multiscale/'
# var = 'tracking_results/ATOMS/var/'
# ratio = 'tracking_results/ATOMS/ratio/'
# var_ratio = 'tracking_results/ATOMS/var_ratio/'
#
# dataset = get_dataset('otb')
#
# for seq in dataset:
#     d_file = '../{}/{}.txt'.format(default, seq.name)
#     ms_file = '../{}/{}.txt'.format(multiscale, seq.name)
#     v_file = '../{}/{}.txt'.format(var, seq.name)
#     r_file = '../{}/{}.txt'.format(ratio, seq.name)
#     vr_file = '../{}/{}.txt'.format(var_ratio, seq.name)
#     # if os.path.isfile(d_file):
#     gt = seq.ground_truth_rect
#     v_bb = np.loadtxt(v_file)
#     r_bb = np.loadtxt(r_file)
#     vr_bb = np.loadtxt(vr_file)
#     d_bb = np.loadtxt(d_file)
#     ms_bb = np.loadtxt(ms_file)
#
#     output_path = './video/{}/'.format(seq.name)
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#
#     for frame_num, frame_path in enumerate(seq.frames):
#         im = cv.imread(seq.frames[frame_num])
#         # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
#         # plt.imshow(im)
#
#         pred_bb =[gt[frame_num], v_bb[frame_num], r_bb[frame_num], vr_bb[frame_num], d_bb[frame_num], ms_bb[frame_num]]
#
#         for i, s in enumerate(pred_bb, start=1):
#             pred_s = s
#             tl = tuple(map(int,[pred_s[0], pred_s[1]]))
#             br = tuple(map(int,[pred_s[0]+pred_s[2], pred_s[1]+pred_s[3]]))
#             col = _tracker_disp_colors[i]
#             cv.rectangle(im, tl, br, col, 2)
#             plt.imshow(im)
#
#         cv.imwrite('{}/{}.jpg'.format(output_path, frame_num), im)
#
# print('done!')

Baseline = '../results/tracking_results/dimp/super_dimp_000'
SuperDiMP = '../results/tracking_results/dimp/super_dimp_no_al_000'
Ours = '../results/tracking_results/rvt/rvt_1_000'

dataset = get_dataset('otb')

sequence = 'Liquor' # Basketball
dataset = [dataset[sequence]]


for seq in dataset:
    gt = seq.ground_truth_rect
    Baseline_file = '{}/{}.txt'.format(Baseline, sequence)
    SuperDiMP_file = '{}/{}.txt'.format(SuperDiMP, sequence)
    Ours_file = '{}/{}.txt'.format(Ours, sequence)

    Baseline = np.loadtxt(Baseline_file)
    SuperDiMP = np.loadtxt(SuperDiMP_file)
    Ours = np.loadtxt(Ours_file)

    output_path = './video/{}/'.format(sequence)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for frame_num, frame_path in enumerate(seq.frames):
        im = cv.imread(seq.frames[frame_num])
        # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # plt.imshow(im)

        pred_bb =[gt[frame_num], Baseline[frame_num], SuperDiMP[frame_num], Ours[frame_num]]

        for i, s in enumerate(pred_bb, start=1):
            pred_s = s
            tl = tuple(map(int,[pred_s[0], pred_s[1]]))
            br = tuple(map(int,[pred_s[0]+pred_s[2], pred_s[1]+pred_s[3]]))
            col = _tracker_disp_colors[i]
            cv.rectangle(im, tl, br, col, 2)
            # plt.imshow(im)
            # plt.show()

        cv.imwrite('{}/{}.jpg'.format(output_path, frame_num), im)

print('done!')