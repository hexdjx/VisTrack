import os
import torch
import numpy as np
from pytracking.evaluation import get_dataset, trackerlist
from tqdm import tqdm
from pytracking.utils.load_text import load_text

trackers = []
# trackers.extend(trackerlist('dimp', 'super_dimp', None, 'SuperDiMP'))  # otb uav nfs lasot
# trackers.extend(trackerlist('fudimp', 'fudimp_ff', None, 'FuDiMP_ff'))  # otb, nfs, uav, lasot
# trackers.extend(trackerlist('fudimp', 'fudimp_awff', None, 'FuDiMP_awff'))  # otb, nfs, uav, lasot
# trackers.extend(trackerlist('fudimp', 'fudimp_awff_att', None, 'FuDiMP_awff_att'))  # otb, nfs, uav, lasot
trackers.extend(trackerlist('fudimp_mu', 'default', None, 'FuDiMP_MU'))  # otb, nfs, uav, lasot

dataset = get_dataset('lasot')
ave_fps = torch.zeros((len(trackers), len(dataset)), dtype=torch.float32)
# ave_fps = torch.zeros(len(dataset), dtype=torch.float32)

for trk_id, trk in enumerate(trackers):
    for seq_id, seq in enumerate(tqdm(dataset)):
        # Load results
        base_results_path = '{}/{}'.format(trk.results_dir, seq.name)

        time_path = '{}_time.txt'.format(base_results_path)
        if os.path.isfile(time_path):
            time_per_frame = torch.tensor(load_text(str(time_path), dtype=np.float64))
        exec_time = time_per_frame.sum()
        fps = len(time_per_frame) / exec_time
        ave_fps[trk_id, seq_id] = fps

    print(trk.display_name, ave_fps[trk_id, :].mean())
