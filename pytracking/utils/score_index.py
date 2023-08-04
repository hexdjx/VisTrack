import torch

# @author Xuedong He
# --------------------------------------------------------------------------
# MOSSE: Peak to Sidelobe Ratio
def PSR(score):
    max_score = score.max()
    psr = (max_score - score.mean()) / (score.std() + 1e-8)
    return psr


# LMCF：Average Peak to Correlation Energy
def APCE(score):
    max_value = score.max()
    min_value = score.min()
    score_min = score - min_value
    score_min_mean = score_min.square().sum() / score.numel()
    apce = torch.square(max_value - min_value) / score_min_mean
    return apce.sqrt()


# our method: Primary and Secondary Peak Mean Energy
def PSME(score, scores_masked):
    max_score1 = score.max()
    max_score2 = scores_masked.max()

    score_mean = (score - score.mean()).square().sum() / score.numel()

    psme = torch.square(max_score1 - max_score2) / score_mean
    return psme.sqrt()
# --------------------------------------------------------------------------