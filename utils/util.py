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
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
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



def compute_all_metrics(pred, gt, gt_multi, device):
    """
    calculate all metrics  aadopted from: HumanMAC (ICCV 2023)

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]  # K=50，50种预测
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    # print(pred.shape)
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()

    gt_multi = torch.from_numpy(gt_multi).to(device)
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)

    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]

    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim=3)
    # we can reuse 'dist' to optimize metrics calculation

    mmfde, _ = dist[:, :-1, -1].min(dim=0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim=2).min(dim=0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim=1).min(dim=0)
    fde, _ = dist[:, -1, -1].min(dim=0)
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde, mmade, mmfde
