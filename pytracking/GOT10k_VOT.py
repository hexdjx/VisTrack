from got10k.trackers import Tracker as GOT_Tracker
from got10k.experiments import ExperimentVOT
import numpy as np
import os
import sys
import argparse


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Sequence, Tracker

parser = argparse.ArgumentParser(description='Run VOT.')
parser.add_argument('--tracker_name', type=str, default='endimp')
parser.add_argument('--tracker_param', type=str, default='endimp_vot18')
parser.add_argument('--run_id', type=int, default=None)
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
args = parser.parse_args()

TrTracker = Tracker(args.tracker_name, args.tracker_param, args.run_id)


class GOT_Tracker(GOT_Tracker):
    def __init__(self):
        super(GOT_Tracker, self).__init__(name='EnDiMP')
        self.tracker = TrTracker.tracker_class(TrTracker.get_parameters())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.initialize(image, box)

    def update(self, image):
        image = np.array(image)
        self.box = self.tracker.track(image)
        return self.box


if __name__ == '__main__':
    # setup tracker
    tracker = GOT_Tracker()

    # run experiments on VOT
    experiment = ExperimentVOT('/media/hexd6/aede3fa6-c741-4516-afe7-4954b8572ac9/907856427856276E/VOT/VOT2018',
                               version=2018, experiments='supervised')
    experiment.run(tracker, visualize=False)

    # report performance
    # experiment.report([tracker.name])
