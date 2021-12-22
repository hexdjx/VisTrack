from pytracking.evaluation.environment import EnvSettings
import os

def local_env_settings():
    settings = EnvSettings()

    base_path = '/home/hexd6/code/Tracking/'
    dataset_path = '/media/hexd6/aede3fa6-c741-4516-afe7-4954b8572ac9/907856427856276E/'

    settings.network_path = dataset_path + 'pretrained_networks'
    # Set your local paths here.
    settings.dataspec_path = base_path + 'VisTrack/ltr/data_specs'
    settings.davis_dir = dataset_path + 'DAVIS/2017'
    settings.got10k_path = dataset_path + 'GOT-10k'
    settings.lasot_path = dataset_path + 'LaSOT/LaSOTBenchmark'
    settings.lasot_extension_subset_path = dataset_path + 'LaSOT_extension_subset'
    settings.nfs_path = dataset_path + 'NFS'
    settings.otb_path = dataset_path + 'OTB100'
    settings.tpl_path = dataset_path + 'TC128'
    settings.trackingnet_path = dataset_path + 'TrackingNet'
    settings.uav_path = dataset_path + 'UAV123'
    settings.youtubevos_dir = dataset_path + 'YouTubeVOS/2018'
    settings.vot_path = dataset_path + 'VOT/2018'
    # result paths
    settings.result_plot_path = base_path + 'VisTrack/pytracking/results/plots/'
    settings.results_path = base_path + 'VisTrack/pytracking/results/tracking_results/'
    settings.segmentation_path = base_path + 'VisTrack/pytracking/results/segmentation_results/'
    settings.tn_packed_results_path = base_path + 'VisTrack/pytracking/results/TrackingNet/'

    return settings

