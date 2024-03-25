import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from einops import rearrange

import models.S_TransformerEncoder as S_Trans_Encoder
import models.PositionEncodings as PositionEncodings


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # nn.embedding  look-up table from_pretrained默认训练过程不更新
            nn.Linear(d_model, d_model),
            Swish(),  # swish激活函数  x*sigmoid(x)  TODO：换成relu
            nn.Linear(d_model, d_model),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 初始化nn.Linear
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        # t = (b) -> (b, dim)
        emb = self.timembedding(t)
        return emb


class PoseSeqEncoder(nn.Module):
    def __init__(self,
                 t_length,
                 joint_num,
                 T_enc_hiddims,
                 S_model_dims,
                 S_trans_enc_num_layers,
                 S_num_heads,
                 S_dim_feedforward,
                 S_dropout_rate,
                 device,
                 posit_encoding_params=(10000, 1),
                 ):
        super().__init__()

        self.joint_num = joint_num
        self.device = device

        self.Temporal_MLP_Encoder = nn.Sequential(
            nn.Linear(t_length * 3, T_enc_hiddims),
            nn.Tanh(),
            nn.Linear(T_enc_hiddims, S_model_dims),
        )

        self.posit_encoder = PositionEncodings.PositionEncodings1D(
            num_pos_feats=S_model_dims,
            temperature=posit_encoding_params[0],
            alpha=posit_encoding_params[1]
        )

        self.Spatial_Attn_Encoder = S_Trans_Encoder.S_TransformerEncoder(
            num_layers=S_trans_enc_num_layers,
            model_dims=S_model_dims,
            num_heads=S_num_heads,
            dim_feedforward=S_dim_feedforward,
            dropout_rate=S_dropout_rate,
            joint_num=joint_num
        )

    def forward(self, pose_seq):
        # pose_seq: (b, t, n, 3)
        # output: (b, n, dim)

        inpu = rearrange(pose_seq, 'b t n d -> b n (t d)')
        out = self.Temporal_MLP_Encoder(inpu)
        S_posit_encoding = self.posit_encoder(self.joint_num).type(torch.float64).to(self.device)
        out, att_weight = self.Spatial_Attn_Encoder(out, S_posit_encoding)
        return out


class Predictor(nn.Module):
    def __init__(self, T,
                 t_his, t_pred, joint_num,
                 T_enc_hiddims,
                 S_model_dims,
                 S_trans_enc_num_layers,
                 S_num_heads,
                 S_dim_feedforward,
                 S_dropout_rate,
                 T_dec_hiddims,
                 fusion_add,
                 device):
        super().__init__()

        self.t_his = t_his
        self.t_pred = t_pred
        self.joint_num = joint_num
        self.fusion_add = fusion_add

        self.Timestep_encoder = TimeEmbedding(T, S_model_dims)
        self.Condition_encoder = PoseSeqEncoder(t_his,
                                                joint_num,
                                                T_enc_hiddims,
                                                S_model_dims,
                                                S_trans_enc_num_layers,
                                                S_num_heads,
                                                S_dim_feedforward,
                                                S_dropout_rate,
                                                device)
        self.Future_encoder = PoseSeqEncoder(t_pred,
                                             joint_num,
                                             T_enc_hiddims,
                                             S_model_dims,
                                             S_trans_enc_num_layers,
                                             S_num_heads,
                                             S_dim_feedforward,
                                             S_dropout_rate,
                                             device)

        self.Decoder = nn.Sequential(
            nn.Linear(S_model_dims if self.fusion_add else S_model_dims*3, T_dec_hiddims),
            nn.Tanh(),
            nn.Linear(T_dec_hiddims, t_pred * 3),
        )

    def forward(self, condition, noisy_future, timestep):
        timestep_feature = self.Timestep_encoder(timestep)  # timestep = (b, ) -> (b, dim)
        condition_feature = self.Condition_encoder(condition)  # condition = (b, t_his, n, 3) -> (b, n, dim)
        future_feature = self.Future_encoder(noisy_future)  # future = (b, t_pred, n, 3) -> (b, n, dim)

        timestep_feature = torch.unsqueeze(timestep_feature, dim=1)
        if self.fusion_add:
            concat_feature = timestep_feature + condition_feature + future_feature  # (b, n, dim)
        else:
            concat_feature = torch.concat([timestep_feature.repeat(1, self.joint_num, 1), condition_feature, future_feature], dim=-1)
        output_noise = self.Decoder(concat_feature)  # (b, n, t_pred*3)
        output_noise = rearrange(output_noise, 'b n (t d) -> b t n d', t=self.t_pred)
        return output_noise


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    T = 1000
    batch_size = 4
    t_his = 10
    t_pred = 25
    joint_num = 22
    timesteps = torch.randint(T, size=(batch_size,))
    condition = torch.randn((batch_size, t_his, joint_num, 3))
    future = torch.randn((batch_size, t_pred, joint_num, 3))

    model = Predictor(T, t_his, t_pred, joint_num,
                      T_enc_hiddims=1024, S_model_dims=256, S_trans_enc_num_layers=2, S_num_heads=8,
                      S_dim_feedforward=512, S_dropout_rate=0, T_dec_hiddims=1024)
    out_feature = model(condition, future, timesteps)
    print(out_feature.shape)
