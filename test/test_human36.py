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
from utils.draw_human36_tool import draw_pic_single
from utils.util import seed_torch, cal_total_model_param
from utils.h36motion3d import Datasets, dim_used
from models.Predictor import Predictor
from models.Diffusion import DDIMSampler


def test_func(gt3d):
    gt3d = gt3d.type(dtype).to(device).contiguous()
    gt3d /= 1000.
    condition = rearrange(gt3d[:, :config.t_his, dim_used], 'b t (c d) -> b t c d', d=3).clone()
    gt_future = rearrange(gt3d[:, config.t_his:, dim_used], 'b t (c d) -> b t c d', d=3).clone()
    noisy_future = torch.randn(size=gt_future.shape, device=device)
    st = time.time()
    sampled_future = sampler(noisy_future, condition)
    print("time =,", time.time()-st)
    sampled_future = rearrange(sampled_future, 'b t c d -> b t (c d)')

    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    pred32 = gt3d.clone()
    pred32[:, config.t_his:config.t_his + config.t_pred, dim_used] = sampled_future[:, :config.t_pred]
    pred32[:, :, index_to_ignore] = pred32[:, :, index_to_equal]
    pred32 = pred32.reshape([-1, config.t_his + config.t_pred, 32, 3])
    gt3d_t = rearrange(gt3d[:, :], 'b t (c d) -> b t c d', d=3).contiguous()
    print('\n', gt3d_t.shape, pred32.shape)

    sample_gt = gt3d_t[5].detach().cpu() * 1000
    sample_pred = pred32[5].detach().cpu() * 1000
    for t_id in range(config.t_his + config.t_pred):
        im_save_dir = os.path.join(config.log_dir, 'vis')
        if not os.path.exists(im_save_dir):
            os.mkdir(im_save_dir)
        draw_pic_single('gt', sample_gt[t_id],
                        os.path.join(im_save_dir, f'gt_{t_id}.png'))
        draw_pic_single('pred', sample_pred[t_id],
                        os.path.join(im_save_dir, f'pred_{t_id}.png'))


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
    test_data_loader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=8,
                                  pin_memory=True)

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

    with torch.no_grad():
        cnt = 0
        for (batch, _) in test_data_loader:
            cnt += 1
            if cnt == 2:
                test_func(batch)
                break
