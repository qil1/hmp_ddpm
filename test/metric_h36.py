import os
import sys
from tqdm import tqdm
from einops import rearrange
import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.opt import Options
from utils.draw_h36_tool import draw_pic_single
from utils.util import seed_torch, cal_total_model_param, AverageMeter, cal_metrics
from utils.dataset_h36m import DatasetH36M
from models.Predictor import Predictor
from models.Diffusion import DDIMSampler


def draw(pose, path):
    pose = pose[:, [0, 2, 1]]
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    for i in range(int(pose.shape[0])):
        x, z, y = pose[i]
        ax.scatter(x, y, z, c='b', s=2)
        ax.text(x, y, z, i, fontsize=6)
    plt.savefig(path)


def test_func():
    generator = dataset.sampling_generator(num_samples=5000, batch_size=1)  # TODO: num_samples full eval, mutimodal
    stats_names = ['APD', 'ADE', 'FDE']
    stats_meter = {x: AverageMeter() for x in stats_names}
    K = 50
    cnt = 0
    for gt3d in generator:  # batch size = 1
        cnt += 1
        gt3d = torch.tensor(gt3d)
        gt3d = gt3d.type(dtype).to(device).contiguous()
        condition = gt3d[:, :config.t_his].clone()
        gt_future = gt3d[:, config.t_his:].clone()
        gt3d_t = gt3d[:, config.t_his:]

        _, t, c, d = gt_future.shape
        stacked_noise_future = torch.randn(size=(K, t, c, d), device=device)
        stacked_condition = condition.repeat(K, 1, 1, 1)
        pred32 = sampler(stacked_noise_future, stacked_condition)

        # if cnt == 5:
        #     pred_t = pred[0][0].cpu()*1000
        #     for t_id in range(pred_t.shape[0]):
        #         im_save_dir = os.path.join(config.log_dir, 'vis')
        #         if not os.path.exists(im_save_dir):
        #             os.mkdir(im_save_dir)
        #         draw(pred_t[t_id], os.path.join(im_save_dir, f'pred_{t_id}.png'))
        pred = torch.unsqueeze(pred32, 1)
        apd, ade, fde = cal_metrics(gt3d_t.cpu(), pred.cpu())
        stats_meter['APD'].update(apd)
        stats_meter['ADE'].update(ade)
        stats_meter['FDE'].update(fde)
        logstr = 'APD: ' + str(stats_meter['APD'].avg) + \
                ', ADE: ' + str(stats_meter['ADE'].avg) + \
                ', FDE: ' + str(stats_meter['FDE'].avg)
        sys.stdout.write(f'\r                cnt={cnt}  '+logstr)
        sys.stdout.flush()
        with open(os.path.join(config.log_dir, 'test_metric.txt'), 'a') as f:
            f.write(logstr+'\n')


if __name__ == "__main__":
    config = Options().parse()
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    config.log_dir = os.path.join(config.log_dir, config.save_dir_name)
    ckpt_path = os.path.join(os.path.join(config.log_dir, 'ckpts'), 'model_best.p')

    '''data'''
    dataset = DatasetH36M('test', config.t_his, config.t_pred)

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
    # sampler = GaussianDiffusionSampler(
    #     model, config.beta_1, config.beta_T, config.T, w=config.w).to(device)
    sampler = DDIMSampler(
        model, config.beta_1, config.beta_T, config.T, w=config.w, device=device).to(device)
    sampler.eval()
    cal_total_model_param([sampler])

    with torch.no_grad():
        test_func()
