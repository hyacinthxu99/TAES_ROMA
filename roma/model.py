"""Channel adaptation around a pretrained reconstruction backbone."""
from torch import nn
from .lora import LoRAInjector


class ROMAModel(nn.Module):
    """Route [batch, time, new_dim] through adapters and a frozen backbone.

    ``backbone`` must already be configured for old_dim input/output channels.
    Load any pretrained backbone weights before constructing this wrapper.
    """

    def __init__(self, backbone, old_dim, new_dim, seq_len, causal_map=None,
                 rank=8, alpha=1.0, adapter_variant="full", device=None, **kwargs):
        super().__init__()
        self.old_dim, self.new_dim, self.seq_len = old_dim, new_dim, seq_len
        self.injector = LoRAInjector(old_dim, new_dim, seq_len, backbone, rank=rank,
                                     alpha=alpha, adapter_variant=adapter_variant, **kwargs)
        self.backbone = self.injector.inject(causal_map, device)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.new_dim):
            raise ValueError(f"Expected [batch, {self.seq_len}, {self.new_dim}], got {tuple(x.shape)}")
        adapted, _ = self.backbone.input_adapter(x)
        reconstructed = self.backbone(adapted, None, None, None)
        return self.backbone.output_adapter(reconstructed)

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
