from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.train_feature_size = 18
    params.feature_stride = 16
    params.image_sample_size = params.train_feature_size*params.feature_stride
    params.search_area_scale = 5
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 2
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = False
    params.augmentation = {}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = False
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True
    params.conf_ths = 0.9
    params.search_area_rescaling_at_occlusion = True

    # ------------------------------------------------------------
    # params.use_target_template = True
    params.use_target_template_fuse = True
    # ------------------------------------------------------------

    # JDTrack_online 18/5
    # JDTrack_online_target_256 16/4 | JDTrack_online_target_384 24/5
    # JDTrack_online_target_fuse.pth.tar
    params.net = NetWithBackbone(net_path='JDTrack/ViT/JDTrack_online_target_fuse.pth.tar', use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    params.use_gt_box = True
    params.plot_iou = True

    return params
