import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class Model(nn.Module):
    """
    DTAAD-style dual TCN-attention baseline for multivariate anomaly detection.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.input_projection = nn.Linear(configs.enc_in, configs.d_model)
        self.short_tcn = nn.Sequential(
            TemporalBlock(configs.d_model, kernel_size=3, dilation=1, dropout=configs.dropout),
            TemporalBlock(configs.d_model, kernel_size=3, dilation=2, dropout=configs.dropout),
        )
        self.long_tcn = nn.Sequential(
            TemporalBlock(configs.d_model, kernel_size=5, dilation=1, dropout=configs.dropout),
            TemporalBlock(configs.d_model, kernel_size=5, dilation=4, dropout=configs.dropout),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_ff),
            nn.GELU() if configs.activation == "gelu" else nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model),
        )
        self.output_projection = nn.Linear(configs.d_model, configs.c_out)
        self.norm = nn.LayerNorm(configs.d_model)

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x_enc - means) / stdev

        hidden = self.input_projection(x)
        conv_input = hidden.transpose(1, 2)
        short = self.short_tcn(conv_input).transpose(1, 2)
        long = self.long_tcn(conv_input).transpose(1, 2)
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        fused = self.fusion(torch.cat([short, long, attended], dim=-1))
        recon = self.output_projection(self.norm(fused + hidden))
        return recon * stdev[:, :, : recon.shape[-1]] + means[:, :, : recon.shape[-1]]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("DTAAD only supports anomaly_detection")
        return self.anomaly_detection(x_enc)
