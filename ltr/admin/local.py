class EnvironmentSettings:
    def __init__(self):
        # Base directory for saving network checkpoints.
        self.workspace_dir = '/media/hexd6/aede3fa6-c741-4516-afe7-4954b8572ac9/907856427856276E/'

        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = self.workspace_dir + 'LaSOT/LaSOTBenchmark'
        self.got10k_dir = self.workspace_dir + 'GOT-10k/train'
        self.trackingnet_dir = self.workspace_dir + 'TrackingNet'
        self.coco_dir = self.workspace_dir + 'COCO'
        self.davis_dir = self.workspace_dir + 'DAVIS'
        self.youtubevos_dir = self.workspace_dir + 'YouTubeVOS'
