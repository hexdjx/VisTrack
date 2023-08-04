import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from pytracking.evaluation import Sequence, Tracker
from ltr.data.image_loader import imwrite_indexed
from pytracking.utils.box_utils import convert_vot_anno_to_rect

import matplotlib.pyplot as plt


# from pytracking.vot_utils.region import vot_float2str

def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    base_results_path = os.path.join(tracker.results_dir, seq.name)
    segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    # def save_vot_bb(file, data):
    #     with open(file, 'w') as f:
    #         for x in data:
    #             if isinstance(x, int):
    #                 f.write("{:d}\n".format(x))
    #             else:
    #                 f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

                # Single-object mode
                # bbox_file = '{}.txt'.format(base_results_path)
                # try:
                #     save_bb(bbox_file, data)
                # except:
                #     save_vot_bb(bbox_file, data)



        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

        elif key == 'segmentation':
            assert len(frame_names) == len(data)
            if not os.path.exists(segmentation_path):
                os.makedirs(segmentation_path)
            for frame_name, frame_seg in zip(frame_names, data):
                imwrite_indexed(os.path.join(segmentation_path, '{}.png'.format(frame_name)), frame_seg)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None):
    """Runs a tracker on a sequence."""

    def _results_exist():
        if seq.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    ##################################################
    # my add
    # plot precision
    def _show_precision(positions, ground_truth, title, max_threshold=50):
        assert positions.shape == ground_truth.shape

        ground_truth = np.array([convert_vot_anno_to_rect(g) for g in ground_truth])
        positions = np.array([convert_vot_anno_to_rect(p) for p in positions])

        distances = np.sqrt(np.square(positions[:, :2] - ground_truth[:, :2]).sum(axis=1))

        precisions = np.zeros([max_threshold, 1])
        for p in range(1, max_threshold + 1):
            precisions[p - 1] = sum(distances < p) / len(distances)
            if p == 20:
                print('precision threshold 20 pix is %.3f' % precisions[p - 1])
        # plt.figure()
        # plt.title('Precisions - ' + title)
        # plt.plot(precisions, 'r-', linewidth=2)
        # plt.xlabel('Threshold')
        # plt.ylabel('Precision')
        # plt.show()

    # plot AUC
    def _show_success(pred_bb, ground_truth, title, max_threshold=1):
        assert pred_bb.shape == ground_truth.shape

        ground_truth = np.array([convert_vot_anno_to_rect(g) for g in ground_truth])
        pred_bb = np.array([convert_vot_anno_to_rect(p) for p in pred_bb])

        tl = np.maximum(pred_bb[:, :2], ground_truth[:, :2])
        br = np.minimum(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, ground_truth[:, :2] + ground_truth[:, 2:] - 1.0)
        sz = np.maximum(br - tl + 1.0, 0)

        # Area
        intersection = sz.prod(axis=1)
        union = pred_bb[:, 2:].prod(axis=1) + ground_truth[:, 2:].prod(axis=1) - intersection
        IOU = intersection / union

        success = np.zeros([100, 1])
        for p in range(1, 101):
            success[p - 1] = sum(IOU > p * 0.01) / len(IOU)
            if p == 50:
                print('success threshold 0.5 is %.3f' % success[p - 1])
        # plt.figure()
        # plt.title('Success - ' + title)
        # plt.plot(success, 'r-', linewidth=2)
        # plt.xlabel('Threshold')
        # plt.ylabel('success')
        # plt.show()

    if debug:
        predict_bb = np.array(output['target_bbox'])
        _show_precision(predict_bb, seq.ground_truth_rect, seq.name)
        _show_success(predict_bb, seq.ground_truth_rect, seq.name)

    ##################################################

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')
