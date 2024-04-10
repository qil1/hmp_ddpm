import os
import random
import numpy as np
import torch
from einops import rearrange


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_total_model_param(model_list):
    total = 0.
    for model in model_list:
        model_para = sum([param.nelement() for param in model.parameters()])
        total += model_para

    print("Number of parameter: %.2fM" % (total / 1e6))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if isinstance(val, int):
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            self.sum += np.sum(np.array(val), keepdims=False)
            self.count += len(val)
            self.avg = self.sum / self.count


def cal_metrics(gt, pred):
    # gt (b, t, c, d)
    # pred (K, b, t, c, d)

    K, b, t, c, d = pred.shape
    diversity_lst = []
    ade_lst = []
    fde_lst = []
    for i in range(b):
        pred_i = pred[:, i]
        dist_diverse = torch.pdist(pred_i.reshape(pred_i.shape[0], -1))
        diversity = dist_diverse.mean().item()
        diversity_lst.append(diversity)

        diff = torch.unsqueeze(gt[i], 0) - pred_i  # (K, t, c, d)
        norm = torch.norm(diff.reshape(pred_i.shape[0], pred_i.shape[1], -1), dim=-1, p=2)
        min_norm = norm.mean(dim=-1).min().item()
        ade_lst.append(min_norm)

        diff = diff[:, -1]
        norm = torch.norm(diff.reshape(pred_i.shape[0], -1), dim=-1, p=2)
        min_norm = norm.min().item()
        fde_lst.append(min_norm)

    return diversity_lst, ade_lst, fde_lst
