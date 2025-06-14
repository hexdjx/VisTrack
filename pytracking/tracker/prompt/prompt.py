from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation
import ltr.data.processing_utils as prutils
import cv2 as cv
import numpy as np
from collections import defaultdict


class ProDiMP(BaseTracker):
    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

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

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # update or not ----------------------------------------------------------
        self.target_prob = torch.unsqueeze(self.target_prob[0], dim=0)
        self.replace_target_prob = self.target_prob

        if self.params.get('prob_update', False) or self.params.get('prob_replace', False):
            self.update_lr = self.params.learning_rate
        # -----------------------------------------------------------------------
        # frame-wise 偏移距离预判目标定位不确定性-------------
        if self.params.get('use_dist_score', False):
            self.disp_threshold = self.params.dispalcement_scale * self.target_sz.prod().sqrt() / 2

            # multi-frame
            self.pos_memory = [self.pos.clone()]

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

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
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw = self.classify_target(test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind, :] + translation_vec

        if self.params.get('use_dist_score', False):
            disp_norm = torch.sqrt(torch.sum((new_pos - self.pos) ** 2))
            if disp_norm > self.disp_threshold:
                flag = 'uncertain'

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind, :], sample_scales[scale_ind], scale_ind,
                                       update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        # ------- UPDATE ------- #
        # if self.params.get('use_dist_score', False):

            # average value
            # disp_norm = 0.0
            # for i_pos in self.pos_memory:
            #     disp_norm += torch.sqrt(torch.sum((i_pos - self.pos) ** 2))

            # disp_norm = disp_norm/len(self.pos_memory)
            # if disp_norm > self.disp_threshold:
            #     flag = 'uncertain'

            # no average
            # for i_pos in self.pos_memory:
            #     disp_norm = torch.sqrt(torch.sum((i_pos - self.pos) ** 2))
            #     if disp_norm > self.disp_threshold:
            #         flag = 'uncertain'
            #         break

            # if len(self.pos_memory) < self.params.dis_frame_num:
            #     self.pos_memory.extend([self.pos.clone()])
            # else:
            #     self.pos_memory.pop(0)
            #     self.pos_memory.extend([self.pos.clone()])

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # self.disp_threshold = self.params.dispalcement_scale * self.target_sz.prod().sqrt() / 2

            # Get train sample
            train_x = test_x[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])

        # ----------------------------------------------------------------------
        if self.params.get('prob_update', False) or self.params.get('prob_replace', False):
            if update_flag:
                self.update_lr = self.params.learning_rate
            else:
                self.update_lr = 0
        # ----------------------------------------------------------------------

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()

        # Visualize and set debug info
        self.search_area_box = torch.cat(
            (sample_coords[scale_ind, [1, 0]], sample_coords[scale_ind, [3, 2]] - sample_coords[scale_ind, [1, 0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        # out = {'target_bbox': output_state, 'score': [max_score, sim]}

        out = {'target_bbox': output_state}
        return out

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

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

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
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        if self.params.get('score_match', False):
            if max_score.item() < self.params.target_not_found_threshold:
                return translation_vec, scale_ind, scores, 'not_found'

        return translation_vec, scale_ind, scores, None

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind, ...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

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
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change',
                                                                                            None))

        # -------------------------------------------------------------------------------------------------
        sample_pos, sample_scales = self.get_sample_location(patch_coords)

        if not self.params.get('init_prob', False):
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0], sample_scales[0])

            if self.params.prompt_style == 'prob':
                target_prob = self.get_target_prob(im_patches[0], target_box)
            else:
                target_prob = self.bbox2mask(target_box)

            if self.params.get('prob_replace', False):
                if self.update_lr > 0:
                    self.replace_target_prob = (1 - self.update_lr) * self.replace_target_prob + self.update_lr * target_prob
                    self.target_prob = target_prob
                else:
                    self.target_prob = self.replace_target_prob
            else:
                if self.params.get('prob_update', False):
                    self.target_prob = (1 - self.update_lr) * self.target_prob + self.update_lr * target_prob
                else:
                    self.target_prob = target_prob

        # if self.params.prompt_style == 'label':
        #     target_center_norm = (pos - sample_pos[0]) / (sample_scales[0] * self.img_support_sz)
        #
        #     for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
        #         ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
        #         center = sz * target_center_norm + 0.5 * ksz_even
        #         target_prob = dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even)
        #         self.target_prob = F.interpolate(target_prob.reshape(-1, *target_prob.shape[-3:]), size=(22, 22),
        #                                      mode='bilinear').to(self.params.device)
        #
        #     # target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0], sample_scales[0])
        #     #
        #     # target_prob = prutils.gaussian_label_function(target_box.cpu().view(-1, 4),
        #     #                                               self.params.output_sigma/self.params.search_area_scale,
        #     #                                               self.params.target_filter_sz,
        #     #                                               self.params.feature_sz, self.params.feature_sz * 16,
        #     #                                               end_pad_if_even=False).to(self.params.device)
        #     # self.target_prob = target_prob.reshape(-1, *target_prob.shape[-3:]).to(self.params.device)
        # -------------------------------------------------------------------------------------------------

        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat, self.target_prob)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)

    def generate_init_samples(self, im: torch.Tensor):
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

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: (
                    (torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend(
                [augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend(
                [augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend(
                [augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend(
                [augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend(
                [augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz,
                                              self.transforms)

        # ----------------------------------------------------------------------------
        if self.params.prompt_style == 'prob':
            p_target_boxes = self.init_target_boxes()
            target_probs = []
            for img, bbox in zip(im_patches, p_target_boxes):
                target_prob = self.get_target_prob(img, bbox)
                target_probs.append(target_prob)

            self.target_prob = torch.cat(target_probs, dim=0)

        # -------------------------------------------------------------------------------------------------
        if self.params.prompt_style == 'mask':
            p_target_boxes = self.init_target_boxes()
            target_probs = []
            for bbox in p_target_boxes:
                target_prob = self.bbox2mask(bbox)
                target_probs.append(target_prob)
            self.target_prob = torch.cat(target_probs, dim=0)

        # if self.params.prompt_style == 'label':
        #
        #     target_probs = []
        #
        #     # method 1
        #     # Center pos in normalized img_coords
        #     self.sigma = (self.params.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * \
        #                  self.params.output_sigma * torch.ones(2)
        #     target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)
        #
        #     ksz_even = torch.Tensor([(self.params.target_filter_sz + 1) % 2, (self.params.target_filter_sz + 1) % 2])
        #     center_pos = self.params.feature_sz * target_center_norm + 0.5 * ksz_even
        #     for T in self.transforms:
        #         sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.params.feature_sz
        #         target_prob = dcf.label_function_spatial(torch.Tensor([self.params.feature_sz,self.params.feature_sz]) , self.sigma, sample_center,
        #                                                  end_pad=ksz_even)
        #         target_prob = F.interpolate(target_prob.reshape(-1, *target_prob.shape[-3:]), size=(22, 22), mode='bilinear')
        #         target_probs.append(target_prob)
        #     self.target_prob = torch.cat(target_probs, dim=0).to(self.params.device)
        #
        #     # method 2
        #     # for bbox in p_target_boxes:
        #     #     target_prob = prutils.gaussian_label_function(bbox.cpu().view(-1, 4),
        #     #                                                   self.params.output_sigma/self.params.search_area_scale,
        #     #                                                   self.params.target_filter_sz,
        #     #                                                   self.params.feature_sz, self.params.feature_sz * 16,
        #     #                                                   end_pad_if_even=False)
        #     #     target_probs.append(target_prob.reshape(-1, *target_prob.shape[-3:]))
        #     #
        #     # self.target_prob = torch.cat(target_probs, dim=0).to(self.params.device)

        # -------------------------------------------------------------------------------------------------

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    # --------------------------------------------------------------------------
    def bbox2mask(self, bbox):
        mask = torch.zeros([self.params.image_sample_size, self.params.image_sample_size]).to(self.params.device)
        x, y, w, h = bbox.tolist()

        x1 = round(x)
        y1 = round(y)
        x2 = round(x + w)
        y2 = round(y + h)
        # Crop target
        mask[y1:y2, x1:x2] = 1.0

        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask.reshape(-1, *mask.shape[-3:]), size=(22, 22), mode='bilinear')

        return mask

    def get_target_prob(self, img, bbox):
        def calHist_mask(im, mask, n_bins=16):

            h, w, c = im.shape

            hist_mask = [
                torch.Tensor(cv.calcHist([im.numpy()], [i], mask.numpy().astype(np.uint8), [16], [0, 256])) for i
                in range(c)]

            counts = torch.stack(hist_mask).squeeze().T

            counts_p = counts / mask.sum()

            bin_width = 256 / n_bins
            bin_indices = torch.floor(im.reshape(-1, c) / bin_width)

            for i in range(n_bins):
                for j in range(c):
                    bin_indices[:, j][i == bin_indices[:, j]] = counts_p[i, j]

            img_p = bin_indices.reshape(h, w, c)

            return img_p

        img = img.permute(1, 2, 0)
        h, w, c = img.shape

        x, y, bw, bh = bbox.tolist()

        bg_mask = torch.ones([h, w], dtype=torch.bool)
        bg_mask[round(y):round(y + bh), round(x):round(x + bw)] = False

        fg_mask = ~bg_mask

        fg_p = calHist_mask(img, fg_mask)

        bg_p = calHist_mask(img, bg_mask)

        P = fg_p / (fg_p + bg_p + 0.01)

        P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)
        P = torch.where(torch.isinf(P), torch.full_like(P, 0), P)

        # output_window = dcf.hann2d(torch.tensor([352, 352]).long(), centered=True)
        # P = P * output_window.squeeze().unsqueeze(-1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(P.numpy())
        # plt.show()

        P = P.permute(2, 0, 1)

        if self.params.get('prompt_up', True):
            P = F.interpolate(P.reshape(-1, *P.shape[-3:]), size=(22, 22), mode='bilinear')
            return P.to(self.params.device)
        else:
            return P.unsqueeze(0).to(self.params.device)
    # --------------------------------------------------------------------------

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

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

    def update_memory(self, sample_x: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos,
                                                         self.init_sample_scale)
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (
                        augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal,
                        augmentation.FlipVertical,
                        augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor(
                [self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1, 4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0], ...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            x = torch.cat([x, F.dropout2d(x[0:1, ...].expand(num, -1, -1, -1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (
                        self.output_sz * self.params.effective_search_area / self.params.search_area_scale).long(),
                                                        centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                     'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor',
                              self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3,
                                         'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale=True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat(
                [self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min() / 3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:, 2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:] / 2) + rand_bb[:, :2]
            init_boxes = torch.cat([new_center - new_sz / 2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1, 4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:, 2] / output_boxes[:, 3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (
                aspect_ratio > 1 / self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind, :]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)

    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))

    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]],
                                       device=self.params.device).view(1, 1, 4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1, 4).cpu(), outputs.detach().view(-1).cpu()

    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                self.params.device).view(1, 1, 4)

        sz_norm = output_boxes[:, :1, 2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient=torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1, 4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale=True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind + 1, ...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1, 4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1, 1, 4)
        sz_norm = center_box[..., 2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist + pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist + sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0, :, 0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:, 0, 1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0, :, 2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:, 0, 3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1, -1, 4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1, -1, 4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(), -1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(), -1), title='Size scores', fig_num=22)

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')


class ProToMP(BaseTracker):
    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

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

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        if self.params.get('prob_update', False) or self.params.get('prob_replace', False):
            self.replace_target_prob = self.target_prob
            self.update_lr = self.params.learning_rate

        # 5帧最大偏移距离预判目标定位不确定性-------------
        if self.params.get('use_dist_score', False):
            self.disp_threshold = self.params.dispalcement_scale * self.target_sz.prod().sqrt() / 2
            # multi-frame
            self.pos_memory = [self.pos.clone()]

        self.logging_dict = defaultdict(list)

        self.target_scales = []
        self.target_not_found_counter = 0

        self.cls_weights_avg = None

        out = {'time': time.time() - tic}
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

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        # Extract classification features
        test_x = self.get_backbone_head_feat(backbone_feat)

        with torch.no_grad():
            test_x = self.net.head.extract_head_feat(test_x, self.target_prob)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, bbox_preds = self.classify_target(test_x)

        translation_vec, scale_ind, s, flag, score_loc = self.localize_target(scores_raw, sample_pos, sample_scales)

        bbox_raw = self.direct_bbox_regression(bbox_preds, sample_coords, score_loc, scores_raw)
        bbox = self.clip_bbox_to_image_area(bbox_raw, image)

        if flag == 'normal' and self.params.get('use_dist_score', False):
            new_pos = bbox[:2].flip(0) + bbox[2:].flip(0) / 2

            disp_norm = torch.sqrt(torch.sum((new_pos - self.pos) ** 2))
            if disp_norm > self.disp_threshold:
                flag = 'uncertain'

        if flag != 'not_found':
            self.pos = bbox[:2].flip(0) + bbox[2:].flip(0) / 2  # [y + h/2, x + w/2]
            self.target_sz = bbox[2:].flip(0)
            self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
            self.target_scales.append(self.target_scale)
        else:
            if self.params.get('search_area_rescaling_at_occlusion', False):
                self.search_area_rescaling()

        # ------- UPDATE ------- #
        # if self.params.get('use_dist_score', False):
        #     disp_norm = 0.0
        #     for i_pos in self.pos_memory:
        #         disp_norm += torch.sqrt(torch.sum((i_pos - self.pos) ** 2))
        #
        #     disp_norm = disp_norm/len(self.pos_memory)
        #     if disp_norm > self.disp_threshold:
        #         flag = 'uncertain'
        #         # break
        #     if len(self.pos_memory) < self.params.dis_frame_num:
        #         self.pos_memory.extend([self.pos.clone()])
        #     else:
        #         self.pos_memory.pop(0)
        #         self.pos_memory.extend([self.pos.clone()])


        update_flag = flag not in ['not_found', 'uncertain']

        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False) and scores_raw.max() > self.params.get(
                'conf_ths', 0.0):

            self.disp_threshold = self.params.dispalcement_scale * self.target_sz.prod().sqrt() / 2

            # Get train sample
            train_x = test_x[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind, :], sample_scales[scale_ind]).to(
                self.params.device)

            # Update the classifier model
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate)

        # ----------------------------------------------------------------------
        if self.params.get('prob_update', False) or self.params.get('prob_replace', False):
            if update_flag:
                self.update_lr = self.params.learning_rate
            else:
                self.update_lr = 0
        # ----------------------------------------------------------------------

        score_map = s[scale_ind, ...]

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        # Visualize and set debug info
        self.search_area_box = torch.cat(
            (sample_coords[0, [1, 0]], sample_coords[0, [3, 2]] - sample_coords[0, [1, 0]] - 1))

        if self.params.get('output_not_found_box', False):
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state,
               'object_presence_score': score_map.max().cpu().item()}

        if self.visdom is not None:
            self.visualize_raw_results(score_map)

        return out

    def visualize_raw_results(self, score_map):
        self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
        self.logging_dict['max_score'].append(score_map.max())
        self.visdom.register(torch.tensor(self.logging_dict['max_score']), 'lineplot', 3, 'Max Score')
        self.debug_info['max_score'] = score_map.max().item()
        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

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

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            # train_samples = self.training_samples[0][:self.num_stored_samples[0], ...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0], :]

            train_feat = self.training_samples[0][:self.num_stored_samples[0], ...]
            test_feat = sample_x

            train_ltrb = self.encode_bbox(target_boxes)
            cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = \
                self.net.head.get_filter_and_features_in_parallel(train_feat, test_feat,
                                                                  num_gth_frames=self.num_gth_frames,
                                                                  train_label=target_labels,
                                                                  train_ltrb_target=train_ltrb)

            # fuse encoder and decoder features to one feature map
            target_scores = self.net.head.classifier(cls_test_feat_enc, cls_weights)

            # compute the final prediction using the output module
            bbox_preds = self.net.head.bb_regressor(bbreg_test_feat_enc, bbreg_weights)

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
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind, ...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None, max_disp

    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
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
        # --target probability-- ##############################################
        sample_pos, sample_scales = self.get_sample_location(patch_coords)

        # if self.params.prompt_style == 'label':
        #     # target_center_norm = (pos - sample_pos[0]) / (sample_scales[0] * self.img_support_sz)
        #     #
        #     # for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
        #     #     ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
        #     #     center = sz * target_center_norm + 0.5 * ksz_even
        #     #     target_prob = dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even)
        #     #     self.target_prob = target_prob.reshape(-1, *target_prob.shape[-3:]).to(self.params.device)
        #
        #     target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0], sample_scales[0])
        #
        #     target_prob = prutils.gaussian_label_function(target_box.cpu().view(-1, 4),
        #                                                   self.params.output_sigma/self.params.search_area_scale,
        #                                                   self.params.target_filter_sz,
        #                                                   self.params.feature_sz, self.params.feature_sz * 16,
        #                                                   end_pad_if_even=False).to(self.params.device)
        #     self.target_prob = target_prob.reshape(-1, *target_prob.shape[-3:]).to(self.params.device)
        # # -------------------------------------------------------------------------------------------------
        #
        # if self.params.prompt_style == 'mask':
        #     target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0], sample_scales[0])
        #     target_prob = self.bbox2mask(target_box)
        #
        #     self.target_prob = F.interpolate(target_prob.reshape(-1, *target_prob.shape[-3:]), size=(18, 18),
        #                                      mode='bilinear').to(self.params.device)

        if self.params.prompt_style == 'prob':
            if not self.params.get('init_prob', False):
                target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0], sample_scales[0])

                target_prob = self.get_target_prob(im_patches[0], target_box)

                if self.params.get('prob_replace', False):
                    if self.update_lr > 0:
                        self.replace_target_prob = (1 - self.update_lr) * self.replace_target_prob + self.update_lr * target_prob
                        self.target_prob = target_prob
                    else:
                        self.target_prob = self.replace_target_prob
                else:
                    if self.params.get('prob_update', False):
                        self.target_prob = (1 - self.update_lr) * self.target_prob + self.update_lr * target_prob
                    else:
                        self.target_prob = target_prob

        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_backbone_head_feat(self, backbone_feat):
        with torch.no_grad():
            return self.net.get_backbone_head_feat(backbone_feat)

    # -----------------------------------------------------------

    def bbox2mask(self, bbox):
        mask = torch.zeros([self.params.image_sample_size, self.params.image_sample_size])
        x, y, w, h = bbox.tolist()

        x1 = round(x)
        y1 = round(y)
        x2 = round(x + w)
        y2 = round(y + h)
        # Crop target
        mask[y1:y2, x1:x2] = 1.0

        # cv.imwrite('./mask.jpg', mask.numpy() * 255)

        # import matplotlib.pyplot as plt
        # plt.figure(10)
        # plt.imshow(mask.cpu().numpy())
        # plt.show()

        return mask.unsqueeze(0).to(self.params.device)  # .unsqueeze(0)

    def get_target_prob(self, img, bbox):
        def calHist_mask(im, mask, n_bins=16):

            h, w, c = im.shape

            hist_mask = [
                torch.Tensor(cv.calcHist([im.numpy()], [i], mask.numpy().astype(np.uint8), [16], [0, 256])) for i
                in range(c)]

            counts = torch.stack(hist_mask).squeeze().T

            counts_p = counts / mask.sum()

            bin_width = 256 / n_bins
            bin_indices = torch.floor(im.reshape(-1, c) / bin_width)

            for i in range(n_bins):
                for j in range(c):
                    bin_indices[:, j][i == bin_indices[:, j]] = counts_p[i, j]

            img_p = bin_indices.reshape(h, w, c)

            return img_p

        img = img.permute(1, 2, 0)
        h, w, c = img.shape

        x, y, bw, bh = bbox.tolist()

        bg_mask = torch.ones([h, w], dtype=torch.bool)
        bg_mask[round(y):round(y + bh), round(x):round(x + bw)] = False

        fg_mask = ~bg_mask

        fg_p = calHist_mask(img, fg_mask)
        bg_p = calHist_mask(img, bg_mask)

        P = fg_p / (fg_p + bg_p + 0.01)

        P = torch.where(torch.isnan(P), torch.full_like(P, 0), P)
        P = torch.where(torch.isinf(P), torch.full_like(P, 0), P)

        output_window = dcf.hann2d(torch.tensor([self.params.image_sample_size, self.params.image_sample_size]).long(), centered=True)

        P = P * output_window.squeeze().unsqueeze(-1)
        P = P.permute(2, 0, 1)

        if self.params.get('prompt_up', True):
            P = F.interpolate(P.reshape(-1, *P.shape[-3:]), size=(self.params.train_feature_size, self.params.train_feature_size), mode='bilinear')
            return P.to(self.params.device)
        else:
            return P.unsqueeze(0).to(self.params.device)

    # -----------------------------------------------------------

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

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: (
                    (torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend(
                [augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz / 2).long().tolist()
            self.transforms.extend(
                [augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in
                 augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend(
                [augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend(
                [augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend(
                [augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz,
                                              self.transforms)

        # ----------------------------------------------------------------------------
        if self.params.prompt_style == 'prob':
            p_target_boxes = self.init_target_boxes()

            target_probs = []
            for img, bbox in zip(im_patches, p_target_boxes):
                target_prob = self.get_target_prob(img, bbox)
                target_probs.append(target_prob)

            self.target_prob = torch.cat(target_probs, dim=0)
        # ----------------------------------------------------------------------------

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

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

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
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

        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0], ...] = x

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind + 1, ...] = y

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples,
                              learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples,
                                                    num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5 * ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

    def update_state(self, new_pos, new_scale=None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_backbone_head_feat(init_backbone_feat)

        # -------------------------------------------------------------------------------------------------
        # if self.params.prompt_style == 'mask':
        #     p_target_boxes = self.init_target_boxes()
        #
        #     target_probs = []
        #     for bbox in p_target_boxes:
        #         target_prob = self.bbox2mask(bbox)
        #
        #         target_prob = F.interpolate(target_prob.reshape(-1, *target_prob.shape[-3:]), size=(18, 18),
        #                                     mode='bilinear').to(self.params.device)
        #         target_probs.append(target_prob)
        #
        #     self.target_prob = torch.cat(target_probs, dim=0)
        #
        # if self.params.prompt_style == 'label':
        #
        #     target_probs = []
        #
        #     # method 1
        #     # Center pos in normalized img_coords
        #     # self.sigma = (self.params.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * \
        #     #              self.params.output_sigma * torch.ones(2)
        #     # target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)
        #     #
        #     # ksz_even = torch.Tensor([(self.params.target_filter_sz + 1) % 2, (self.params.target_filter_sz + 1) % 2])
        #     # center_pos = self.params.feature_sz * target_center_norm + 0.5 * ksz_even
        #     # for T in self.transforms:
        #     #     sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.params.feature_sz
        #     #     target_prob = dcf.label_function_spatial(torch.Tensor([self.params.feature_sz,self.params.feature_sz]) , self.sigma, sample_center,
        #     #                                              end_pad=ksz_even)
        #     #
        #     #     target_prob = target_prob.reshape(-1, *target_prob.shape[-3:])
        #     #     target_probs.append(target_prob)
        #     # self.target_prob = torch.cat(target_probs, dim=0).to(self.params.device)
        #
        #     # method 2
        #     p_target_boxes = self.init_target_boxes()
        #
        #     for bbox in p_target_boxes:
        #         target_prob = prutils.gaussian_label_function(bbox.cpu().view(-1, 4),
        #                                                       self.params.output_sigma/self.params.search_area_scale,
        #                                                       self.params.target_filter_sz,
        #                                                       self.params.feature_sz, self.params.feature_sz * 16,
        #                                                       end_pad_if_even=False)
        #         target_probs.append(target_prob.reshape(-1, *target_prob.shape[-3:]))
        #
        #     self.target_prob = torch.cat(target_probs, dim=0).to(self.params.device)

        # ------------------------------
        with torch.no_grad():
            x = self.net.head.extract_head_feat(x, self.target_prob)
        # ---------------------------

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1] * num)
            x = torch.cat([x, F.dropout2d(x[0:1, ...].expand(num, -1, -1, -1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = getattr(self.net.head.filter_predictor, 'filter_size', 1)
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (
                        self.output_sz * self.params.effective_search_area / self.params.search_area_scale).long(),
                                                        centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target labels for the different augmentations
        self.init_target_labels(TensorList([x]))

        self.num_gth_frames = target_boxes.shape[0]

        if hasattr(self.net.head.filter_predictor, 'num_gth_frames'):
            self.net.head.filter_predictor.num_gth_frames = self.num_gth_frames

        self.init_memory(TensorList([x]))

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')
