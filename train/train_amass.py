import os
import sys
from tqdm import tqdm
from einops import rearrange
import time
import json
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from utils.opt import Options
from utils.torch import to_cpu
from utils.util import seed_torch, cal_total_model_param
from utils.amass3d import Datasets
from models.Predictor import Predictor
from models.Diffusion import GaussianDiffusionTrainer


def train_func(epoch):
    train_loss = 0
    total_num_sample = 0
    t_s = time.time()
    with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
        for gt3d in tqdmDataLoader:
            gt3d = gt3d.type(dtype).to(device).contiguous()
            gt3d /= 1000.
            condition = gt3d[:, :config.t_his, dataset.joint_used].clone()
            future = gt3d[:, config.t_his:, dataset.joint_used].clone()
            # print(condition.shape, future.shape)
            loss = trainer(future, condition).sum() / 1000.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            total_num_sample += 1

            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch": epoch,
                "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                "loss": loss.item()
            })
    dt = time.time() - t_s
    scheduler.step()
    train_loss /= total_num_sample
    log_str = f"[train epoch {i}] total time: {dt}, average loss: {train_loss}"
    print(log_str)
    with open(os.path.join(config.log_dir, 'train_log.txt'), 'a') as f:
        f.write(log_str+'\n')


if __name__ == "__main__":
    config = Options().parse()
    seed_torch(config.seed)

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

    config.log_dir = os.path.join(config.log_dir, config.save_dir_name)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(os.path.join(config.log_dir, 'ckpts')):
        os.mkdir(os.path.join(config.log_dir, 'ckpts'))
    with open(os.path.join(config.log_dir, 'config.txt'), 'w') as f:
        config_str = json.dumps(vars(config), indent=4, ensure_ascii=False)
        f.write(config_str)

    '''data'''
    dataset = Datasets(config, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

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
    trainer = GaussianDiffusionTrainer(
        model, config.beta_1, config.beta_T, config.T).to(device)
    trainer.train()
    cal_total_model_param([trainer])

    '''optimizer'''
    optimizer = optim.AdamW(trainer.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.5)

    for i in range(0, config.num_epoch):
        print("epoch:", i)
        train_func(i)

        with to_cpu(model):
            ckpt_path = os.path.join(os.path.join(config.log_dir, 'ckpts'), 'model_%04d.p' % i)
            model_cp = {'model_dict': model.state_dict()}
            pickle.dump(model_cp, open(ckpt_path, 'wb'))
