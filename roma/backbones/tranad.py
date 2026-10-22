import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.input_projection = nn.Linear(configs.enc_in, configs.d_model)
        self.position = PositionalEncoding(configs.d_model, configs.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        self.decoder_phase1 = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU() if configs.activation == "gelu" else nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.c_out),
        )
        self.focus_projection = nn.Linear(configs.enc_in, configs.d_model)
        self.decoder_phase2 = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU() if configs.activation == "gelu" else nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.c_out),
        )

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x_enc - means) / stdev

        hidden = self.encoder(self.position(self.input_projection(x)))
        recon1 = self.decoder_phase1(hidden)
        focus = torch.abs(recon1.detach() - x)
        hidden2 = self.encoder(self.position(self.input_projection(x) + self.focus_projection(focus)))
        recon2 = self.decoder_phase2(hidden2)
        return recon2 * stdev[:, :, : recon2.shape[-1]] + means[:, :, : recon2.shape[-1]]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("TranAD only supports anomaly_detection")
        return self.anomaly_detection(x_enc)
