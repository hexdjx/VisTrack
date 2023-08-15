from pytracking.external.got10k.trackers import Tracker as GOT_Tracker
from pytracking.external.got10k.experiments import ExperimentOTB, ExperimentUAV123, ExperimentNfS, ExperimentTColor128, \
    ExperimentGOT10k
import numpy as np
import os
import sys
import argparse
import torch

from pytracking.evaluation.environment import env_settings
from pytracking.external.refine_modules.refine_module import RefineModule
from pytracking.external.refine_modules.utils import get_axis_aligned_bbox, bbox_clip

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

parser = argparse.ArgumentParser(description='Run GOT10K.')
parser.add_argument('--tracker_name', type=str, default='endimp')
parser.add_argument('--tracker_param', type=str, default='endimp')
parser.add_argument('--run_id', type=int, default=None)
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
args = parser.parse_args()

TrTracker = Tracker(args.tracker_name, args.tracker_param, args.run_id)


class GOT_Tracker(GOT_Tracker):
    def __init__(self):
        super(GOT_Tracker, self).__init__(name='EnDiMP')  # GOT_Tracker
        self.tracker = TrTracker.tracker_class(TrTracker.get_parameters())

    def init(self, image, box):
        image = np.array(image)

        init_info = {}
        init_info['init_bbox'] = box

        self.tracker.initialize(image, init_info)

    def update(self, image):
        image = np.array(image)
        outputs = self.tracker.track(image)
        pred_bbox = outputs['target_bbox']

        return pred_bbox


# using AlphaRefine to refine base tracker
class GOT_Tracker_AR(GOT_Tracker):
    def __init__(self):
        super(GOT_Tracker, self).__init__(name='EnDiMP_AR')
        self.tracker = TrTracker.tracker_class(TrTracker.get_parameters())

        # create Refinement module
        self.RF_module = RefineModule(os.path.join(env_settings().network_path, 'SEcmnet_ep0040-c.pth.tar'), selector=0)

    def init(self, image, box):
        image = np.array(image)

        # Initialize base tracker
        init_info = {}
        init_info['init_bbox'] = box

        self.tracker.initialize(image, init_info)

        # Initialize refine tracker
        self.im_H, self.im_W, _ = image.shape
        cx, cy, w, h = get_axis_aligned_bbox(np.array(box))
        gt_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        self.RF_module.initialize(image, np.array(gt_bbox))

    def update(self, image):
        image = np.array(image)
        outputs = self.tracker.track(image)
        pred_bbox = outputs['target_bbox']

        # refine tracking results
        pred_bbox = self.RF_module.refine(image, np.array(pred_bbox))

        # post-processing
        x1, y1, w, h = pred_bbox.tolist()
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.im_H, self.im_W))  # add boundary and min size limit
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / self.tracker.base_target_sz.prod())

        # update tracker's state with refined result
        self.tracker.pos = new_pos.clone()
        self.tracker.target_sz = new_target_sz
        self.tracker.target_scale = new_scale

        return pred_bbox


if __name__ == '__main__':
    # setup tracker
    tracker = GOT_Tracker_AR()  # GOT_Tracker()

    Experiments = [
        (ExperimentOTB, env_settings().otb_path),
        (ExperimentUAV123, env_settings().uav_path),
        (ExperimentNfS, env_settings().nfs_path),
        (ExperimentTColor128, env_settings().tpl_path),
        (ExperimentGOT10k, env_settings().got10k_path)
    ]

    # run experiments using GOT-10k
    for exps, data_root in Experiments:
        if os.path.basename(data_root) == "GOT-10k":
            experiment = exps(data_root,
                              subset='test',
                              result_dir=os.path.join(env_settings().got_results_path, 'results'),
                              report_dir=os.path.join(env_settings().got_results_path, 'reports'))
        elif os.path.basename(data_root) == "NFS":
            experiment = exps(data_root,
                              fps=30,
                              result_dir=os.path.join(env_settings().got_results_path, 'results'),
                              report_dir=os.path.join(env_settings().got_results_path, 'reports'))
        else:
            experiment = exps(data_root,
                              result_dir=os.path.join(env_settings().got_results_path, 'results'),
                              report_dir=os.path.join(env_settings().got_results_path, 'reports'))

        experiment.run(tracker, visualize=False)

        # report performance
        experiment.report([tracker.name])
