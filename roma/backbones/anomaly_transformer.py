import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim * n_heads != d_model:
            raise ValueError("d_model must be divisible by n_heads")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.sigma_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        distances = torch.arange(seq_len).view(1, -1) - torch.arange(seq_len).view(-1, 1)
        self.register_buffer("distance", distances.float().abs().view(1, 1, seq_len, seq_len), persistent=False)

    def forward(self, x: torch.Tensor):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        series = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        series = self.dropout(series)

        sigma = F.softplus(self.sigma_proj(x)).transpose(1, 2).unsqueeze(-1) + 1e-5
        distance = self.distance[:, :, :length, :length]
        prior = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(series, v)
        out = out.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(out), series, prior


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seq_len: int, dropout: float, activation: str):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        attn_out, series, prior = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, series, prior


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.association_lambda = getattr(configs, "association_lambda", 0.1)

        self.value_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, configs.seq_len, configs.d_model))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    seq_len=configs.seq_len,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        self.last_series = []
        self.last_prior = []

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x_enc - means) / stdev

        x = self.value_embedding(x) + self.position_embedding[:, : x.shape[1], :]
        series_list = []
        prior_list = []
        for layer in self.layers:
            x, series, prior = layer(x)
            series_list.append(series)
            prior_list.append(prior)
        self.last_series = series_list
        self.last_prior = prior_list

        out = self.projection(self.norm(x))
        return out * stdev[:, :, : out.shape[-1]] + means[:, :, : out.shape[-1]]

    def association_loss(self) -> torch.Tensor:
        if not self.last_series:
            return torch.tensor(0.0, device=self.position_embedding.device)
        losses = []
        for series, prior in zip(self.last_series, self.last_prior):
            series = series.clamp_min(1e-8)
            prior = prior.detach().clamp_min(1e-8)
            losses.append(torch.mean(torch.sum(series * (torch.log(series) - torch.log(prior)), dim=-1)))
        return torch.stack(losses).mean()

    def compute_loss(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        recon = self.anomaly_detection(x_enc)
        return F.mse_loss(recon, x_enc) + self.association_lambda * self.association_loss()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("AnomalyTransformerAD only supports anomaly_detection")
        return self.anomaly_detection(x_enc)
