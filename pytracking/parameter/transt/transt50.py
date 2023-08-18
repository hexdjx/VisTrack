from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    # Scale penalty
    params.PENALTY_K = 0

    # Window influence
    params.WINDOW_INFLUENCE = 0.49

    # Exemplar size
    params.EXEMPLAR_SIZE = 128

    # Instance size
    params.INSTANCE_SIZE = 256

    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net = NetWithBackbone(net_path='transt.pth',
                                 use_gpu=params.use_gpu)
    return params
