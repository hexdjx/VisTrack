import torch
from pytracking import dcf

# --my add-- #########################################################################################
# MOSSE: Peak to Sidelobe Ratio
def PSR(self, score, sample_scale, output_sz):
    sz = score.shape[-2:]
    max_score, max_disp = dcf.max2d(score)
    max_disp = max_disp.float().cpu().view(-1)
    target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
            output_sz / self.img_support_sz)
    tneigh_top = max(round(max_disp[0].item() - target_neigh_sz[0].item() / 2), 0)
    tneigh_bottom = min(round(max_disp[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
    tneigh_left = max(round(max_disp[1].item() - target_neigh_sz[1].item() / 2), 0)
    tneigh_right = min(round(max_disp[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
    score_crop = score[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right]
    psr = (max_score - score_crop.mean()) / (score_crop.std() + 1e-8)

    return psr

# LMCF：average peak-to correlation energy
def APCE(self, score):
    max_value = score.max()
    min_value = score.min()
    score_min = score - min_value
    score_min_mean = score_min.square().sum() / score.numel()
    apce = torch.square(max_value - min_value) / score_min_mean
    apce = apce.view(-1)

    return apce.cpu().numpy()[0]

def crop_score(self, score, max_disp, target_neigh_sz, sz):
    tneigh_top = max(round(max_disp[0].item() - target_neigh_sz[0].item() / 2), 0)
    tneigh_bottom = min(round(max_disp[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
    tneigh_left = max(round(max_disp[1].item() - target_neigh_sz[1].item() / 2), 0)
    tneigh_right = min(round(max_disp[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])

    score_tmp = score.clone()
    score_max = score_tmp[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right]

    score[..., tneigh_top:tneigh_bottom, tneigh_left:tneigh_right] = 0

    return score_max, score

# Paper: Improving model drift for robust object tracking
# Primary and Secondary Peak Mean Difference Ratio
def PSMD(self, score, sample_scale, output_sz):
    sz = score.shape[-2:]
    s_mean = score.mean()

    max_score1, max_disp1 = dcf.max2d(score)
    max_disp1 = max_disp1.float().cpu().view(-1)
    target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
            output_sz / self.img_support_sz)

    _, scores_masked = self.crop_score(score, max_disp1, target_neigh_sz, sz)

    max_score2, _ = dcf.max2d(scores_masked)

    psmd = (max_score1 - s_mean) / torch.abs(max_score2 - s_mean)
    return psmd.cpu().numpy()[0]

def IoU(self, pos1, pos2, sz):
    rec1 = torch.cat((pos1 - sz / 2, pos1 + sz / 2), dim=-1)
    rec2 = torch.cat((pos2 - sz / 2, pos2 + sz / 2), dim=-1)
    l_max = max(rec1[0], rec2[0])
    r_min = min(rec1[2], rec2[2])
    t_max = max(rec1[1], rec2[1])
    b_min = min(rec1[3], rec2[3])

    intersection = (b_min - t_max) * (r_min - l_max)
    union = 2 * sz.prod()
    iou = intersection.clamp(0) / (union - intersection)
    return iou

#  detection for distractor
def localize_advanced2(self, scores, sample_pos, sample_scales):
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
    score = scores_hn[scale_ind:scale_ind + 1, ...]

    # previous translation vector
    prev_target_vec = (self.pos - sample_pos[scale_ind, :]) / ((self.img_support_sz / output_sz) * sample_scale)
    pre_translation_vec = prev_target_vec * (self.img_support_sz / output_sz) * sample_scale

    # first peak translation vector
    target_disp1 = max_disp1 - score_center
    translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

    if max_score1.item() < self.params.target_not_found_threshold:
        return translation_vec1, scale_ind, scores_hn, 'not_found'

    # Mask out target neighborhood
    target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (
            output_sz / self.img_support_sz)

    # the primary peak
    score_copy = score.clone()

    # the first peak
    score_max1, scores_masked = self.crop_score(score_copy, max_disp1, target_neigh_sz, sz)
    # psr1 = (max_score1 - score_max1.mean()) / (score_max1.std() + 1e-8)

    max_score2, max_disp2 = dcf.max2d(scores_masked)
    max_disp2 = max_disp2.float().cpu().view(-1)

    # the second peak
    score_max2, _ = self.crop_score(scores_masked, max_disp2, target_neigh_sz, sz)
    # psr2 = (max_score2 - score_max2.mean()) / (score_max2.std() + 1e-8)

    if max_score1.item() > self.params.distractor_threshold and max_score2.item() < self.params.target_not_found_threshold:
        return translation_vec1, scale_ind, scores_hn, 'normal'

    # if self.params.get('iou_detection', False):
    #     iou = self.IoU(max_disp1, max_disp2, target_neigh_sz)
    #     if iou > 0.0:  # and max_score2.item() > self.params.target_not_found_threshold:
    #         return translation_vec1, scale_ind, scores_hn, 'hard_negative'

    # if self.params.get('ratio_with_psr', False):
    #     w_ratio = (psr2 * max_score2) / (psr1 * max_score1)
    # else:
    #     w_ratio = max_score2 / max_score1
    #
    # max_ratio = 1 - w_ratio.item()

    # max value position
    target_disp2 = max_disp2 - score_center
    disp_norm1 = torch.sqrt(torch.sum((target_disp1 - prev_target_vec) ** 2))
    disp_norm2 = torch.sqrt(torch.sum((target_disp2 - prev_target_vec) ** 2))

    # distance maximum range
    dis_range = torch.sqrt(target_neigh_sz.square().sum()) / 2
    dis_flag1 = disp_norm1 < dis_range and disp_norm2 < dis_range
    dis_flag2 = disp_norm1 < dis_range and disp_norm2 < 2 * dis_range
    if max_score1.item() > self.params.hard_negative_threshold and dis_flag1:
        return translation_vec1, scale_ind, scores_hn, 'normal'
    if max_score1.item() > self.params.hard_negative_threshold and dis_flag2:
        return translation_vec1, scale_ind, scores_hn, 'hard_negative'

    # dis_ratio = (dis_range + disp_norm1) / torch.abs(disp_norm1 - disp_norm2)
    # self.s_index.append(t_ratio.item())
    # score map index
    # self.score_index["max_score1"] = max_score1
    # self.score_index["pos1"] = pos_max1
    # self.score_index["psr1"] = psr1
    # # self.score_index["apce1"] = apce1
    #
    # self.score_index["max_score2"] = max_score2
    # self.score_index["pos2"] = pos_max2
    # self.score_index["psr2"] = psr2
    # self.score_index["apce2"] = apce2
    #
    # self.score_index['psmd'] = psmd
    # self.score_index['det'] = det
    # self.score_index["psr_ratio"] = psr_ratio

    # Handle the different cases
    # if max_ratio > self.params.hard_negative_threshold and dis_ratio.item() > self.params.hard_negative_threshold:
    #     return translation_vec1, scale_ind, scores_hn, 'hard_negative'
    return pre_translation_vec, scale_ind, scores_hn, 'uncertain'

######################################################################################################