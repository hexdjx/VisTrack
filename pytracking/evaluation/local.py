import sys
from pytracking.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    if sys.platform == 'linux':
        # linux path
        base_path = '/home/ad/code/VisTrack/'
        dataset_path = '/home/ad/datasets/'  # /media/dell/A5388FC7CFA41BE5/
    else:  # 'win32'
        # win path
        base_path = 'E:/code/Tracking/VisTrack/'
        dataset_path = 'E:/datasets/'

    settings.network_path = dataset_path + 'networks'
    settings.dataspec_path = base_path + '/ltr/data_specs'

    settings.dataset_path = dataset_path
    settings.davis_dir = dataset_path + 'DAVIS/2017'
    settings.got10k_path = dataset_path + 'GOT-10k'
    settings.lasot_path = dataset_path + 'LaSOT'
    settings.lasot_extension_subset_path = dataset_path + 'LaSOT_extension_subset'
    settings.nfs_path = dataset_path + 'NFS'
    settings.otb_path = dataset_path + 'OTB100'
    settings.tpl_path = dataset_path + 'TC128'
    settings.trackingnet_path = dataset_path + 'TrackingNet'
    settings.uav_path = dataset_path + 'UAV123'
    settings.youtubevos_dir = dataset_path + 'YouTubeVOS/2018'
    settings.vot_path = dataset_path + 'VOT2018'

    # result paths
    settings.base_result_path = base_path + 'pytracking/results/'
    settings.result_plot_path = base_path + 'pytracking/results/plots/'
    settings.results_path = base_path + 'pytracking/results/tracking_results/'
    settings.segmentation_path = base_path + 'pytracking/results/segmentation_results/'
    settings.tn_packed_results_path = base_path + 'pytracking/results/TrackingNet/'
    settings.got_results_path = base_path + 'pytracking/results/GOT-10k/'
    settings.vot18_results_path = base_path + 'pytracking/results/VOT2018/'
    settings.vot20_results_path = base_path + 'pytracking/results/VOT2020/'

    return settings
