
import torch
from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    # basic parameters
    params.out_scale = 0.001
    params.exemplar_sz = 127
    params.instance_sz = 255
    params.context = 0.5
    params.search_area_scale = 2

    # inference parameters
    params.scale_step = 1.0375
    params.scale_num = 3

    params.scale_lr = 0.59
    params.scale_penalty = 0.9745
    params.window_influence = 0.176
    params.response_sz = 17
    params.response_up = 16
    params.total_stride = 8

    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net_path = 'D:/Tracking/Datasets/networks/siamfc_alexnet_e50.pth'

    return params
