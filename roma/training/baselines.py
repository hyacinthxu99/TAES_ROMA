"""Same-source full fine-tuning and fixed alignment baselines."""
import torch
from torch import nn
from torch.nn import functional as F

class TrainablePretrainAligner(nn.Module):
    """Fine-tune every pretrained parameter plus linear dimension projections."""
    def __init__(self, base_model, base_dims, new_dims):
        super().__init__()
        self.base_model = base_model
        self.input_projection = nn.Linear(new_dims, base_dims)
        self.output_projection = nn.Linear(base_dims, new_dims)
        with torch.no_grad():
            for layer in (self.input_projection, self.output_projection):
                layer.weight.zero_()
                layer.bias.zero_()
                for i in range(min(layer.in_features, layer.out_features)):
                    layer.weight[i, i] = 1
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, x, *args):
        output = self.base_model(self.input_projection(x), None, None, None)
        if isinstance(output, tuple):
            output = output[0]
        return self.output_projection(output)


class DirectPretrainAligner(nn.Module):
    """Run a fixed base model by padding or truncating target entity dimensions."""

    def __init__(self, base_model: nn.Module, base_dims: int, new_dims: int, padding_value: float = 0.0):
        super().__init__()
        self.base_model = base_model
        self.base_dims = int(base_dims)
        self.new_dims = int(new_dims)
        self.padding_value = float(padding_value)
        for param in self.base_model.parameters():
            param.requires_grad = False

    @property
    def strategy(self) -> str:
        if self.new_dims > self.base_dims:
            return "variable_increase_truncate_input_pad_output"
        if self.new_dims < self.base_dims:
            return "variable_decrease_pad_input_truncate_output"
        return "same_dimension_direct_pretrain"

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        aligned_input = self.align_input(x_enc)
        outputs = self.base_model(aligned_input, None, None, None)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return self.align_output(outputs, x_enc)

    def align_input(self, x: torch.Tensor) -> torch.Tensor:
        return self._resize(x, self.base_dims)

    def align_output(self, outputs: torch.Tensor, original_input: torch.Tensor) -> torch.Tensor:
        return self._resize(outputs, self.new_dims)

    def _resize(self, x, width):
        if x.shape[-1] >= width:
            return x[..., :width]
        return F.pad(x, (0, width - x.shape[-1]), value=self.padding_value)

