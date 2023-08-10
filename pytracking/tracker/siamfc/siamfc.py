
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2

from pytracking.features.preprocessing import numpy_to_torch, sample_patch
from pytracking.tracker.base import BaseTracker


def naive_corr(x, kernel):
    """group conv2d to calculate cross neck, fast version
    """
    batch = kernel.size()[0]
    bx, c, h, w = x.size()
    x = x.view(-1, batch * c, h, w)
    out = F.conv2d(x, kernel, groups=batch)
    out = out.view(bx, -1, out.size()[-2], out.size()[-1])
    return out


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Net(nn.Module):

    def __init__(self, backbone):
        super(Net, self).__init__()
        self.backbone = backbone


class SiamFC(BaseTracker):
    multiobj_mode = 'parallel'

    def initialize(self, image, info: dict) -> dict:
        state = info['init_bbox']

        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        alex_net = Net(AlexNet())

        # load checkpoint if provided
        alex_net.load_state_dict(torch.load(self.params.net_path, map_location='cpu'))

        self.net = alex_net.to(self.params.device)

        # Time initialization
        tic = time.time()

        # Get target position and size
        # Get position and size
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # create hanning window
        self.upscale_sz = self.params.response_up * self.params.response_sz
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        self.scale_factors = self.params.scale_step ** np.linspace(
            -(self.params.scale_num // 2),
            self.params.scale_num // 2, self.params.scale_num)

        # exemplar and search sizes
        context = self.params.context * torch.sum(self.target_sz)
        self.z_sz = torch.round(torch.sqrt(torch.prod(self.target_sz + context))) * torch.ones(2)
        self.x_sz = self.z_sz * self.params.search_area_scale

        exemplar_sz = self.params.exemplar_sz
        self.exemplar_sz = torch.Tensor([exemplar_sz, exemplar_sz] if isinstance(exemplar_sz, int) else exemplar_sz)

        instance_sz = self.params.instance_sz
        self.instance_sz = torch.Tensor([instance_sz, instance_sz] if isinstance(instance_sz, int) else instance_sz)

        im_patch, _ = sample_patch(numpy_to_torch(image), self.pos, self.z_sz, self.exemplar_sz)

        with torch.no_grad():
            self.kernel = self.net.backbone(self.preprocess_image(im_patch))

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:

        im_patchs = []
        for s in self.scale_factors:
            im_patch, _ = sample_patch(numpy_to_torch(image), self.pos, self.x_sz * s, self.instance_sz)
            im_patchs.append(im_patch)
        im_patchs = torch.concat(im_patchs, dim=0)

        with torch.no_grad():
            x = self.net.backbone(self.preprocess_image(im_patchs))

        responses = naive_corr(x, self.kernel)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            r, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC) for r in responses])

        responses[:self.params.scale_num // 2] *= self.params.scale_penalty
        responses[self.params.scale_num // 2 + 1:] *= self.params.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16

        response = (1 - self.params.window_influence) * response + \
                   self.params.window_influence * self.hann_window

        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * self.params.total_stride / self.params.response_up
        disp_in_image = disp_in_instance * self.x_sz[0].item() * self.scale_factors[scale_id] / self.params.instance_sz
        self.pos += disp_in_image

        # update target size
        scale = (1 - self.params.scale_lr) * 1.0 + self.params.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        new_state = np.array([
            self.pos[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.pos[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        out = {'target_bbox': new_state.tolist()}
        return out

    def preprocess_image(self, im: torch.Tensor, image_format='rgb',
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Normalize the image with the mean and standard deviation used by the network."""
        _mean = torch.Tensor(mean).view(1, -1, 1, 1)
        _std = torch.Tensor(std).view(1, -1, 1, 1)

        if image_format in ['rgb', 'bgr']:
            im = im / 255

        if image_format in ['bgr', 'bgr255']:
            im = im[:, [2, 1, 0], :, :]
        im -= _mean
        im /= _std

        if self.params.use_gpu:
            im = im.cuda()

        return im
