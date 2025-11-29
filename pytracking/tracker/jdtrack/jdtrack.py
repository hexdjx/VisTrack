
from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
from ltr.models.layers import activation

from collections import defaultdict


class JDTrack(BaseTracker):

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.params.net.initialize()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        self._mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
        self._std = torch.Tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        tfs = self.params.get('train_feature_size', 18)
        self.feature_sz = torch.Tensor([tfs, tfs] if isinstance(tfs, (int, float)) else tfs)

        stride = self.params.get('feature_stride', 16)
        self.train_img_sample_sz = torch.Tensor([tfs * stride, tfs * stride])

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        ksz = 1
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)

        # Extract and transform sample
        im_patches = self.generate_init_samples(im)

        self.init_classifier(im_patches)

        self.logging_dict = defaultdict(list)

        self.target_scales = []
        self.target_not_found_counter = 0

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                   self.target_scale * self.params.scale_factors,
                                                                   self.img_sample_sz)
        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds = self.mix_classify_target(im_patches)

        translation_vec, scale_ind, s, flag, score_loc = self.localize_target(scores_raw, sample_pos, sample_scales)

        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)

        if flag != 'not_found':
            self.pos = bbox[:2].flip(0) + bbox[2:].flip(0) / 2  # [y + h/2, x + w/2]
            self.target_sz = bbox[2:].flip(0)
            self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
            self.target_scales.append(self.target_scale)
        else:
            if self.params.get('search_area_rescaling_at_occlusion', False):
                self.search_area_rescaling()


        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']

        if update_flag and self.params.get('update_classifier', False) and scores_raw.max() > self.params.get('conf_ths', 0.0):
            # Get train sample
            train_x = im_patches[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])

            if self.params.get('use_target_template_fuse', False):
                self.target_boxs[1] = target_box.view(1,-1).cuda()

            train_y = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(
                self.params.device)

            # Update the classifier model
            self.update_memory(TensorList([train_x]), train_y, target_box)

        score_map = s[scale_ind, ...]

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        out = {'target_bbox': new_state.tolist(),
               'object_presence_score': score_map.max().cpu().item()}

        return out

    def clip_bbox_to_image_area(self, bbox, image, minwidth=10, minheight=10):
        H, W = image.shape[:2]
        x1 = max(0, min(bbox[0], W - minwidth))
        y1 = max(0, min(bbox[1], H - minheight))
        x2 = max(x1 + minwidth, min(bbox[0] + bbox[2], W))
        y2 = max(y1 + minheight, min(bbox[1] + bbox[3], H))
        return torch.Tensor([x1, y1, x2 - x1, y2 - y1])

    def encode_bbox(self, bbox):
        stride = self.params.get('feature_stride')
        output_sz = self.params.get('image_sample_size')

        shifts_x = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shifts_y = torch.arange(
            0, output_sz, step=stride,
            dtype=torch.float32, device=bbox.device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        xs, ys = locations[:, 0], locations[:, 1]

        xyxy = torch.stack([bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2],
                            bbox[:, 1] + bbox[:, 3]], dim=1)

        l = xs[:, None] - xyxy[:, 0][None]
        t = ys[:, None] - xyxy[:, 1][None]
        r = xyxy[:, 2][None] - xs[:, None]
        b = xyxy[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

        reg_targets_per_im = reg_targets_per_im / output_sz

        sz = output_sz // stride
        nb = bbox.shape[0]
        reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)

        return reg_targets_per_im

    def direct_bbox_regression(self, bbox_preds, sample_coords, score_loc, scores_raw):
        shifts_x = torch.arange(
            0, self.img_sample_sz[0], step=16,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, self.img_sample_sz[1], step=16,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + 16 // 2
        xs, ys = locations[:, 0], locations[:, 1]
        s1, s2 = scores_raw.shape[2:]
        xs = xs.reshape(s1, s2)
        ys = ys.reshape(s1, s2)

        ltrb = bbox_preds.permute(0, 1, 3, 4, 2)[0, 0].cpu() * self.train_img_sample_sz[[0, 1, 0, 1]]
        xs1 = xs - ltrb[:, :, 0]
        xs2 = xs + ltrb[:, :, 2]
        ys1 = ys - ltrb[:, :, 1]
        ys2 = ys + ltrb[:, :, 3]
        sl = score_loc.int()

        x1 = xs1[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[
            0, 1]
        y1 = ys1[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[
            0, 0]
        x2 = xs2[sl[0], sl[1]] / self.img_sample_sz[1] * (sample_coords[0, 3] - sample_coords[0, 1]) + sample_coords[
            0, 1]
        y2 = ys2[sl[0], sl[1]] / self.img_sample_sz[0] * (sample_coords[0, 2] - sample_coords[0, 0]) + sample_coords[
            0, 0]
        w = x2 - x1
        h = y2 - y1

        x1_ = xs1[sl[0], sl[1]]
        y1_ = ys1[sl[0], sl[1]]
        x2_ = xs2[sl[0], sl[1]]
        y2_ = ys2[sl[0], sl[1]]
        self.crop_bbox = [x1_,y1_,x2_, y2_]

        return torch.Tensor([x1, y1, w, h])

    def search_area_rescaling(self):
        if len(self.target_scales) > 0:
            min_scales, max_scales, max_history = 2, 30, 60
            self.target_not_found_counter += 1
            num_scales = max(min_scales, min(max_scales, self.target_not_found_counter))
            target_scales = torch.tensor(self.target_scales)[-max_history:]
            target_scales = target_scales[
                target_scales >= target_scales[-1]]  # only boxes that are bigger than the `not found`
            target_scales = target_scales[-num_scales:]  # look as many samples into past as not found endures.
            self.target_scale = torch.mean(target_scales)  # average bigger boxes from the past

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5 * (sample_coord[:, :2] + sample_coord[:, 2:] - 1)
        sample_scales = ((sample_coord[:, 2:] - sample_coord[:, :2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
            self.img_support_sz / (2 * self.feature_sz)

    def mix_classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            train_samples = self.training_samples[0][:self.num_stored_samples[0], ...].unsqueeze(1).cuda()
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...].unsqueeze(1).cuda()
            train_ltrbs = self.train_ltrbs[0][:self.num_stored_samples[0], ...].unsqueeze(1).cuda()

            if self.num_stored_samples[0] == 1:
                image_list = torch.cat([train_samples, train_samples, sample_x.unsqueeze(1).cuda()], dim=0)
                target_labels = torch.cat([target_labels, target_labels], dim=0)
                train_ltrbs = torch.cat([train_ltrbs, train_ltrbs], dim=0)
            else:
                image_list = torch.cat([train_samples, sample_x.unsqueeze(1).cuda()], dim=0)

            if self.params.get('use_target_template_fuse', False): # double target template
                train_bbs = torch.cat(self.target_boxs, dim=0)
                bbreg_weights, bbreg_test_feat_enc = self.net.feature_extractor(image_list.cuda(),
                                                                                target_labels,
                                                                                train_ltrbs,
                                                                                train_bbs)

            elif self.params.get('use_target_template', False): # single target template & not update target template
                bbreg_weights, bbreg_test_feat_enc = self.net.feature_extractor(image_list.cuda(),
                                                                                target_labels,
                                                                                train_ltrbs,
                                                                                self.init_target_box)
            else:
                bbreg_weights, bbreg_test_feat_enc = self.net.feature_extractor(image_list.cuda(),
                                                                                target_labels,
                                                                                train_ltrbs)

            # compute the final prediction using the output module
            target_scores = self.net.classifier(bbreg_test_feat_enc, bbreg_weights)
            bbox_preds = self.net.bb_regressor(bbreg_test_feat_enc, bbreg_weights)

        return target_scores, bbox_preds

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1, 1, score_filter_ksz, score_filter_ksz)
            scores = F.conv2d(scores.view(-1, 1, *scores.shape[-2:]), kernel, padding=score_filter_ksz // 2).view(
                scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1) / 2

        if self.output_window is not None:
            scores *= self.output_window

        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        if max_score.item() < self.params.target_not_found_threshold:
            return translation_vec, scale_ind, scores, 'not_found', max_disp

        return translation_vec, scale_ind, scores, None, max_disp

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2

        scores_hn = scores
        if self.output_window is not None:
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)

        scale_ind = scale_ind.cpu()

        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found', max_disp1
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
                output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])

        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind, :]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1 - prev_target_vec) ** 2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2 - prev_target_vec) ** 2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative', max_disp2
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain', max_disp1

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative', max_disp1

        return translation_vec1, scale_ind, scores_hn, 'normal', max_disp1

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change',
                                                                                            None))
        self.im_patch = im_patches

        im_patches = im_patches / 255
        im_patches -= self._mean
        im_patches /= self._std

        return patch_coords, im_patches

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz,
                                              self.transforms)
        im_patches = im_patches / 255
        im_patches -= self._mean
        im_patches /= self._std

        return im_patches

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos,
                                                         self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0], :] = init_target_boxes
        return init_target_boxes

    def init_online_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] // 16 + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] // 16 + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1 / 4)
        self.sigma = (
                             self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(
            2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center,
                                                               end_pad=ksz_even)

    def init_memory(self, train_x: TensorList, target_boxes: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        self.num_stored_samples = self.num_init_samples.copy()

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

        self.train_ltrbs = TensorList(
            [x.new_zeros(self.params.sample_memory_size, 4, x.shape[2] // 16, x.shape[3] // 16) for x in train_x])

        for tl, bbox in zip(self.train_ltrbs, target_boxes):
            tl[:bbox.shape[0], ...] = self.encode_bbox(bbox)

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box):
        replace_ind = [1]

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y

        for ltrbs_memory, ind in zip(self.train_ltrbs, replace_ind):
            ltrbs_memory[ind:ind + 1, ...] = self.encode_bbox(target_box.view(-1, 4))

        self.num_stored_samples += 1

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5 * ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier(self, im_patches):

        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        self.output_window = None
        if self.params.get('window_output', False):
            self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_online_target_labels(TensorList([im_patches]))

        self.num_gth_frames = target_boxes.shape[0]

        self.init_memory(TensorList([im_patches]), TensorList([target_boxes]))

        if self.params.get('use_target_template_fuse', False):
            self.target_boxs = [target_boxes, target_boxes]

        if self.params.get('use_target_template', False):
            self.init_target_box = target_boxes


