import torch

class cos_similarity:
    def __call__(self, x1, x2):
        # sim = 1 - F.cosine_similarity(x1, x2, dim=1)
        # # a = torch.clamp(sim, min=0.0)
        # return torch.mean(torch.clamp(sim, min=0.0))
        x1_norm = torch.norm(x1,dim=-1)
        x2_norm = torch.norm(x2,dim=-1)
        cosine_d = torch.sum(x1_norm * x2_norm, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]
