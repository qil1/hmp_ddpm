import os
import sys
from tqdm import tqdm
from einops import rearrange
import time
import json
import pickle
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.opt import Options
from utils.util import seed_torch, cal_total_model_param, AverageMeter, cal_metrics
from utils.h36motion3d import Datasets, dim_used
from models.Predictor import Predictor
from models.Diffusion import DDIMSampler


def test_func():
    stats_names = ['APD', 'ADE', 'FDE']
    stats_meter = {x: AverageMeter() for x in stats_names}
    K = 50
    cnt = 0
    for (gt3d, _) in test_data_loader:
        cnt += 1
        gt3d = gt3d.type(dtype).to(device).contiguous()
        gt3d /= 1000.
        condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d', d=3).clone()
        gt_future = rearrange(gt3d[:, config.t_his:, dim_used], 'b t (c d) -> b t c d', d=3).clone()
        noisy_future = torch.randn(size=gt_future.shape, device=device)

        stacked_sampled_future = None
        for k in range(K):
            sampled_future = sampler(noisy_future, condition)
            if stacked_sampled_future is None:
                stacked_sampled_future = torch.unsqueeze(sampled_future, 0)
            else:
                stacked_sampled_future = torch.cat([stacked_sampled_future,
                                                    torch.unsqueeze(sampled_future, 0)], dim=0)
        # print(stacked_sampled_future.shape)
        # stacked_gt_future = gt_future.repeat(K, 1, 1, 1, 1)
        # print(stacked_gt_future.shape)
        apd, ade, fde = cal_metrics(gt_future.cpu(), stacked_sampled_future.cpu())
        stats_meter['APD'].update(apd)
        stats_meter['ADE'].update(ade)
        stats_meter['FDE'].update(fde)
        logstr = 'APD: ' + str(stats_meter['APD'].avg) + \
                 ', ADE: ' + str(stats_meter['ADE'].avg) + \
                 ', FDE: ' + str(stats_meter['FDE'].avg)
        sys.stdout.write(f'\r                cnt={cnt}  ' + logstr)
        sys.stdout.flush()
        with open(os.path.join(config.log_dir, 'test_metric.txt'), 'a') as f:
            f.write(logstr + '\n')


if __name__ == "__main__":
    config = Options().parse()
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    config.log_dir = os.path.join(config.log_dir, config.save_dir_name)
    if str.isnumeric(config.iter):
        model_name = 'model_%04d.p' % int(config.iter)
    else:
        model_name = 'model_best.p'
    ckpt_path = os.path.join(os.path.join(config.log_dir, 'ckpts'), model_name)

    '''data'''
    dataset = Datasets(opt=config, split=2, actions=config.test_act)
    print(len(dataset))
    test_data_loader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=8,
                                  pin_memory=True, drop_last=False)

    '''model'''
    model = Predictor(config.T, config.t_his, config.t_pred, config.joint_num,
                      T_enc_hiddims=config.T_enc_hiddims,
                      S_model_dims=config.S_model_dims,
                      S_trans_enc_num_layers=config.S_trans_enc_num_layers,
                      S_num_heads=config.S_num_heads,
                      S_dim_feedforward=config.S_dim_feedforward,
                      S_dropout_rate=config.S_dropout_rate,
                      T_dec_hiddims=config.T_dec_hiddims,
                      fusion_add=config.fusion_add,
                      device=device)
    model_cp = pickle.load(open(ckpt_path, "rb"))
    model.load_state_dict(model_cp['model_dict'])
    print(f"loaded {ckpt_path}")
    model.eval()
    sampler = DDIMSampler(
        model, config.beta_1, config.beta_T, config.T, w=config.w, device=device).to(device)
    sampler.eval()
    cal_total_model_param([sampler])

    # with torch.no_grad():
    #     cnt = 0
    #     for (batch, _) in test_data_loader:
    #         cnt += 1
    #         if cnt == 5:
    #             test_func(batch)
    #             break
    with torch.no_grad():
        test_func()
