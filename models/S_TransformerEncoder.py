import torch
import torch.nn as nn


class My_ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, attn_pdrop):
        super(My_ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(attn_pdrop)

    def forward(self, q, k, v, attn_mask):

        attn_score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        if attn_mask is not None:
            attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class My_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_pdrop):
        super(My_MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads
        self.d_model = d_model

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = My_ScaledDotProductAttention(self.d_k, attn_pdrop)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        outputs = self.linear(attn)

        return outputs, attn_weights


class My_TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            model_dims,
            num_heads,
            dim_feedforward,
            dropout_rate,
            joint_num=22,
    ):
        super(My_TransformerEncoderLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dim_feedforward = dim_feedforward
        self.model_dims = model_dims

        self.attn = My_MultiHeadAttention(
            d_model=model_dims,
            n_heads=num_heads,
            attn_pdrop=dropout_rate
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear1 = nn.Linear(model_dims, self.dim_feedforward)
        self.linear2 = nn.Linear(self.dim_feedforward, self.model_dims)
        self.norm1 = nn.BatchNorm1d(joint_num, eps=1e-9)
        self.norm2 = nn.BatchNorm1d(joint_num, eps=1e-9)
        # self.norm1 = nn.LayerNorm(model_dims, eps=1e-5)
        # self.norm2 = nn.LayerNorm(model_dims, eps=1e-5)

    def forward(
            self,
            x
    ):
        att_out, att_weights = self.attn(
            Q=x,
            K=x,
            V=x,
            attn_mask=None
        )
        norm_att = self.dropout(att_out) + x
        norm_att = self.norm1(norm_att)

        out = self.linear1(norm_att)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.dropout(out) + norm_att
        out = self.norm2(out)

        return out, att_weights


class S_TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers,
            model_dims,
            num_heads,
            dim_feedforward,
            dropout_rate,
            joint_num=22
    ):
        super(S_TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dim_feedforward = dim_feedforward
        self.model_dims = model_dims
        self.num_heads = num_heads

        self.embedding_layer = nn.Linear(model_dims, model_dims)
        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = My_TransformerEncoderLayer(
                model_dims=self.model_dims,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_rate=self.dropout_rate,
                joint_num=joint_num
            )
            self.encoder_layers.append(layer)

    def forward(
            self,
            x,
            posit_encoding,
    ):
        x = self.embedding_layer(x)
        x = x + posit_encoding

        for i in range(self.num_layers):
            x, att_weights = self.encoder_layers[i](
                x
            )
        return x, att_weights
