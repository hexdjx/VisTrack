import sys

class EnvironmentSettings:
    def __init__(self):
        # Base directory for saving network checkpoints.
        if sys.platform == 'linux':
            # linux path
            self.workspace_dir = '/home/ad/datasets/'  # linux /media/dell/A5388FC7CFA41BE5/ /home/ad/datasets/

        else:  # 'win32'
            # win path
            self.workspace_dir = 'D:/Tracking/Datasets/'  # windows

        self.tensorboard_dir = self.workspace_dir + 'tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + 'networks/'
        self.lasot_dir = self.workspace_dir + 'LaSOT/'
        self.got10k_dir = self.workspace_dir + 'GOT-10k/train'
        self.trackingnet_dir = self.workspace_dir + 'TrackingNet'
        self.coco_dir = self.workspace_dir + 'COCO'
        self.davis_dir = self.workspace_dir + 'DAVIS'
        self.youtubevos_dir = self.workspace_dir + 'YouTubeVOS'
