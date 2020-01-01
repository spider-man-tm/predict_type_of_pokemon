import torch.nn.functional as F


def criterion(logit, truth, weight=None):
    logit, truth = logit.float(), truth.float()
    loss = F.binary_cross_entropy(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()

    return loss