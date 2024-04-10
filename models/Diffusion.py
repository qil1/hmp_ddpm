import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from models.Predictor import Predictor


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, condition):
        """
        Algorithm 1.
        x_0 (B, t_pred, N, 3); labels (B, t_his, N, 3)
        model的输入有三个带噪图像x_t (B, t_pred, N, 3); t (B,); labels (B, t_his, N, 3)
        t是随机生成的，数值范围为[0, T)
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)  # B个[0,T)的随机数
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(condition, x_t, t), noise, reduction='none')  # 求均方误差，reduction='none'
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        """
        采样
        """
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())  # 0.0001 ~ 0.02
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        """
        预测q(x_t-1|x-t)的均值 mean
        """
        assert x_t.shape == eps.shape  # x_t和epsilon形状相同，epsilon为预测的噪声
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps  # 提取t时刻的两个系数并转为x_t形状

    def p_mean_variance(self, x_t, t, condition):
        # below: only log_variance is used in the KL computations
        var = self.posterior_var
        # var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(condition, x_t, t)
        nonEps = self.model(torch.zeros_like(condition).to(condition.device), x_t, t)  # no_guidance
        eps = (1. + self.w) * eps - self.w * nonEps  # w控制guidance的程度，参考文献Classifier-Free Diffusion Guidance

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, condition):
        """
        Algorithm 2.  labels是x_T这个batch的样本对应的标签
        X_T (B T_h N 3)
        condition (B T_f N 3)
        """
        x_t = x_T  # X_T是标准正态分布的完全随机噪声
        for time_step in reversed(range(self.T)):
            sys.stdout.write('\rtimestep: ' + str(time_step))
            sys.stdout.flush()
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step  # 第time_step个时刻
            mean, var = self.p_mean_variance(x_t=x_t, t=t, condition=condition)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return x_0  # torch.clip(x_0, -1, 1)  # 让数值在[-1, 1]范围内


class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0., ddim_timesteps=100, ddim_eta=1.0, device=torch.device('cpu')):
        super().__init__()

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())  # 0.0001 ~ 0.02
        alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        self.device = device

        self.model = model
        self.noise_steps = T
        self.ddim_timesteps = ddim_timesteps
        self.w = w
        self.ddim_eta = ddim_eta

        self.ddim_timestep_seq = np.asarray(
            list(range(0, self.noise_steps+1, self.noise_steps // self.ddim_timesteps)))[1:] - 1

        self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_seq[:-1])

        if self.ddim_timestep_seq[0] != 0:
            self.ddim_timestep_seq = np.append(np.array([0]), self.ddim_timestep_seq)
            self.ddim_timestep_prev_seq = np.append(np.array([0]), self.ddim_timestep_prev_seq)

    def forward(self, x_T, condition):
        """
        X_T (B T_h N 3)
        """
        x_t = x_T  # torch.randn((sample_num, self.motion_size[0], self.motion_size[1])).to(self.device)

        for i in reversed(range(0, len(self.ddim_timestep_seq))):
            sys.stdout.write('\rtimestep: ' + str(i))
            sys.stdout.flush()
            # print(self.ddim_timestep_seq[i], self.ddim_timestep_prev_seq[i])

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_seq[i]
            prev_t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * self.ddim_timestep_prev_seq[i]

            alpha_bar = extract(self.alphas_bar, t, x_t.shape)  # self.alphas_bar[t][:, None, None, None]
            if self.ddim_timestep_seq[i] == 0:
                alpha_bar_prev = x_t.new_ones(x_t.shape, dtype=torch.long)
            else:
                alpha_bar_prev = extract(self.alphas_bar, prev_t, x_t.shape)  # self.alphas_bar[prev_t][:, None, None, None]

            predicted_noise = self.model(condition, x_t, t)

            predicted_x0 = (x_t - torch.sqrt((1. - alpha_bar)) * predicted_noise) / torch.sqrt(alpha_bar)

            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))

            pred_dir_xt = torch.sqrt(1 - alpha_bar_prev - sigmas_t**2) * predicted_noise
            x_prev = torch.sqrt(alpha_bar_prev) * predicted_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x_t)

            x_t = x_prev

        return x_t


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0")

    T = 1000
    batch_size = 4
    t_his = 10
    t_pred = 25
    joint_num = 22

    model = Predictor(T, t_his, t_pred, joint_num,
                      T_enc_hiddims=1024, S_model_dims=256, S_trans_enc_num_layers=2, S_num_heads=8,
                      S_dim_feedforward=512, S_dropout_rate=0, T_dec_hiddims=1024, device=device)

    trainer = GaussianDiffusionTrainer(
        model, 1e-4, 0.02, 1000).to("cuda:0")

    condition = torch.randn((batch_size, t_his, joint_num, 3), device=device)
    clean_future = torch.randn((batch_size, t_pred, joint_num, 3), device=device)
    loss = trainer(clean_future, condition).sum() / 1000.
    print(loss)
