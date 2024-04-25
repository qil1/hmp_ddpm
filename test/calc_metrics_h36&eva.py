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
from utils.util import seed_torch, cal_total_model_param, AverageMeter, compute_all_metrics
from utils.dataset_h36m_multimodal import DatasetH36M_multi, get_multimodal_gt_full
from utils.dataset_humaneva_multimodal import DatasetHumanEva_multi
from models.Predictor import Predictor
from models.Diffusion import DDIMSampler


# def draw(pose, path):
#     pose = pose[:, [0, 2, 1]]
#     plt.figure()
#     ax = plt.subplot(111, projection='3d')
#     ax.grid(False)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_xlim3d([-1000, 1000])
#     ax.set_ylim3d([-1000, 1000])
#     ax.set_zlim3d([-1000, 1000])
#
#     for i in range(int(pose.shape[0])):
#         x, z, y = pose[i]
#         ax.scatter(x, y, z, c='b', s=2)
#         ax.text(x, y, z, i, fontsize=6)
#     plt.savefig(path)


# def test_func():
#     generator = dataset.sampling_generator(num_samples=5000, batch_size=1)
#     stats_names = ['APD', 'ADE', 'FDE']
#     stats_meter = {x: AverageMeter() for x in stats_names}
#     K = 50
#     cnt = 0
#     for gt3d in generator:  # batch size = 1
#         cnt += 1
#         gt3d = torch.tensor(gt3d)
#         gt3d = gt3d.type(dtype).to(device).contiguous()
#         condition = gt3d[:, :config.t_his].clone()
#         gt_future = gt3d[:, config.t_his:].clone()
#         gt3d_t = gt3d[:, config.t_his:]
#
#         _, t, c, d = gt_future.shape
#         stacked_noise_future = torch.randn(size=(K, t, c, d), device=device)
#         stacked_condition = condition.repeat(K, 1, 1, 1)
#         pred32 = sampler(stacked_noise_future, stacked_condition)
#
#         # if cnt == 5:
#         #     pred_t = pred[0][0].cpu()*1000
#         #     for t_id in range(pred_t.shape[0]):
#         #         im_save_dir = os.path.join(config.log_dir, 'vis')
#         #         if not os.path.exists(im_save_dir):
#         #             os.mkdir(im_save_dir)
#         #         draw(pred_t[t_id], os.path.join(im_save_dir, f'pred_{t_id}.png'))
#         pred = torch.unsqueeze(pred32, 1)
#         apd, ade, fde = cal_metrics(gt3d_t.cpu(), pred.cpu())
#         stats_meter['APD'].update(apd)
#         stats_meter['ADE'].update(ade)
#         stats_meter['FDE'].update(fde)
#         logstr = 'APD: ' + str(stats_meter['APD'].avg) + \
#                 ', ADE: ' + str(stats_meter['ADE'].avg) + \
#                 ', FDE: ' + str(stats_meter['FDE'].avg)
#         sys.stdout.write(f'\r                cnt={cnt}  '+logstr)
#         sys.stdout.flush()
#         with open(os.path.join(config.log_dir, 'test_metric.txt'), 'a') as f:
#             f.write(logstr+'\n')

def compute_stats(multimodal_dict, cfg):  # mutimodal, full eval
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def get_prediction(data):
        traj_np = data[..., 1:, :]  # .transpose([0, 2, 3, 1])
        traj = torch.tensor(traj_np, device=device, dtype=torch.float64)
        # traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        condition = traj[:, :cfg.t_his, :, :]
        noise = torch.randn([condition.shape[0], cfg.t_pred, condition.shape[2], 3], dtype=torch.float64, device=device)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]
        # mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')

        sampled_motion = sampler(noise, condition)
        traj_est = traj
        traj_est[:, cfg.t_his:, :, :] = sampled_motion
        traj_est = traj_est.reshape([traj_est.shape[0], traj_est.shape[1], -1])
        # traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: {y: AverageMeter() for y in ['hmp_ddpm']} for x in stats_names}

    K = 50
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5168, 125, 48] in h36m
            pred = pred[:, :, cfg.t_his:, :]
            print(pred.shape)
            print(gt_group.shape)
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to(device)
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to(device)
            except:
                pass
            # pred [50, 5168s, 100, 48]
            for j in range(0, num_samples):
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                                                                        gt_group[j][np.newaxis, ...],
                                                                        traj_gt_arr[j], device)
                stats_meter['APD']['hmp_ddpm'].update(apd.cpu())
                stats_meter['ADE']['hmp_ddpm'].update(ade.cpu())
                stats_meter['FDE']['hmp_ddpm'].update(fde.cpu())
                stats_meter['MMADE']['hmp_ddpm'].update(mmade.cpu())
                stats_meter['MMFDE']['hmp_ddpm'].update(mmfde.cpu())
            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                print(str_stats)
                with open(os.path.join(config.log_dir, 'test_metric.txt'), 'a') as f:
                    f.write(str_stats + '\n')
            pred = []



if __name__ == "__main__":
    config = Options().parse()
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    config.log_dir = os.path.join(config.log_dir, config.save_dir_name)
    ckpt_path = os.path.join(os.path.join(config.log_dir, 'ckpts'), 'model_best.p')

    '''data'''
    # dataset = DatasetH36M('test', config.t_his, config.t_pred)

    if config.dataset == 'h36m':
        dataset_multi_test = DatasetH36M_multi('test', config.t_his, config.t_pred,
                    multimodal_path='./data/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz',
                    data_candi_path='./data/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz')
    else:
        dataset_multi_test = DatasetHumanEva_multi('test', config.t_his, config.t_pred,
                multimodal_path='./data/humaneva_multi_modal/t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz',
                data_candi_path='./data/humaneva_multi_modal/data_candi_t_his15_t_pred60_skiprate15.npz')
    multimodal_dict = get_multimodal_gt_full(dataset_multi_test, config)

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
        compute_stats(multimodal_dict, config)
