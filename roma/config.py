"""Model-side defaults, independent of datasets and experiment routing."""
from dataclasses import dataclass


@dataclass
class BackboneConfig:
    enc_in: int
    seq_len: int = 30
    c_out: int | None = None
    task_name: str = "anomaly_detection"
    pred_len: int = 0
    d_model: int = 32
    d_ff: int = 8
    e_layers: int = 2
    n_heads: int = 4
    dropout: float = .1
    activation: str = "gelu"
    top_k: int = 3
    num_kernels: int = 6
    embed: str = "timeF"
    freq: str = "h"
    label_len: int = 48
    factor: int = 1
    association_lambda: float = .1

    def __post_init__(self):
        if self.c_out is None:
            self.c_out = self.enc_in
        if min(self.enc_in, self.seq_len, self.d_model, self.d_ff, self.e_layers, self.n_heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.task_name != "anomaly_detection" or self.c_out != self.enc_in:
            raise ValueError("This draft supports equal-width anomaly reconstruction only")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
