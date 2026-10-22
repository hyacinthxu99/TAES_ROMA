"""Linear-layer LoRA injection and portable adapter weight serialization."""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .adapters import InputAdapter, InputAdapterBase, InputAdapterLinear


class LoRAInjector:
    """Freeze a backbone, attach LoRA hooks, and register channel adapters.

    Adapters are registered on ``target_model``. Use :class:`ROMAModel` to route
    tensors through them; attaching the adapters alone does not change the
    backbone's top-level forward method.
    """

    def __init__(self, old_dim, new_dim, seq_len, target_model, rank=8, alpha=1.0,
                 adapter_variant="full", gnn_hidden_dim=4, gnn_out_dim=4,
                 gnn_layers=1, adapter_query_mode="factorized"):
        if not all(isinstance(value, int) and value > 0 for value in (old_dim, new_dim, seq_len, rank)):
            raise ValueError("old_dim, new_dim, seq_len and rank must be positive integers")
        if adapter_variant not in ("full", "wo_gcn", "linear"):
            raise ValueError("adapter_variant must be full, wo_gcn or linear")
        self.target_model = target_model
        self.old_dim, self.new_dim, self.seq_len = old_dim, new_dim, seq_len
        self.rank, self.alpha = rank, float(alpha)
        self.beta = self.alpha / rank
        self.adapter_variant = adapter_variant
        self.gnn_hidden_dim, self.gnn_out_dim = gnn_hidden_dim, gnn_out_dim
        self.gnn_layers, self.adapter_query_mode = gnn_layers, adapter_query_mode
        self.lora_modules = {}
        self.adapters = {}

    def inject(self, causal_map=None, device=None):
        if self.lora_modules or hasattr(self.target_model, "input_adapter"):
            raise RuntimeError("This backbone already has an adapter")
        modules = [(name, module) for name, module in self.target_model.named_modules()
                   if isinstance(module, nn.Linear)]
        if any(hasattr(module, "lora_A") or hasattr(module, "lora_B") for _, module in modules):
            raise RuntimeError("This backbone already has LoRA parameters")
        if self.adapter_variant == "full":
            if causal_map is None or tuple(causal_map.shape) != (self.new_dim, self.new_dim):
                raise ValueError("full adapter requires a [new_dim, new_dim] graph")
        device = torch.device(device) if device is not None else next(self.target_model.parameters()).device
        self.target_model.to(device)
        for name, module in modules:
            a = nn.Parameter(torch.randn(self.rank, module.in_features, device=device) * .01)
            b = nn.Parameter(torch.zeros(module.out_features, self.rank, device=device))
            module.register_parameter("lora_A", a)
            module.register_parameter("lora_B", b)
            module.register_forward_hook(self._make_lora_hook())
            self.lora_modules[name] = (a, b)
        for name, parameter in self.target_model.named_parameters():
            if "lora" not in name:
                parameter.requires_grad = False
        self.target_model.input_adapter = self.build_input_adapter(causal_map, device)
        self.target_model.output_adapter = nn.Linear(self.old_dim, self.new_dim).to(device)
        self.adapters = {"input_adapter": self.target_model.input_adapter,
                         "output_adapter": self.target_model.output_adapter}
        return self.target_model.to(device)

    def build_input_adapter(self, causal_map, device):
        if self.adapter_variant == "full":
            return InputAdapter(input_dim=self.new_dim, output_dim=self.old_dim,
                seq_len=self.seq_len, causal_map=causal_map, gnn_hidden=self.gnn_hidden_dim,
                gnn_out=self.gnn_out_dim, gnn_layers=self.gnn_layers,
                query_projection_mode=self.adapter_query_mode, device=device).to(device)
        cls = InputAdapterBase if self.adapter_variant == "wo_gcn" else InputAdapterLinear
        return cls(input_dim=self.new_dim, output_dim=self.old_dim,
                   seq_len=self.seq_len, device=device).to(device)

    def _make_lora_hook(self):
        def hook(module, inputs, output):
            delta = self.beta * F.linear(F.linear(inputs[0], module.lora_A), module.lora_B)
            return output + delta
        return hook

    def _metadata(self):
        return dict(format_version=1, old_dim=self.old_dim, new_dim=self.new_dim,
                    seq_len=self.seq_len, rank=self.rank, alpha=self.alpha,
                    adapter_variant=self.adapter_variant,
                    gnn_hidden_dim=self.gnn_hidden_dim, gnn_out_dim=self.gnn_out_dim,
                    gnn_layers=self.gnn_layers, adapter_query_mode=self.adapter_query_mode)

    def save_lora_only(self, save_dir, identifier="adapter"):
        """Save LoRA and adapter tensors; the frozen backbone is not included."""
        if not self.adapters:
            raise RuntimeError("Inject the model before saving plugin weights")
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"plugin_{identifier}.pt"
        state = {"metadata": self._metadata(),
                 "lora": {name: (a.detach().cpu().clone(), b.detach().cpu().clone())
                          for name, (a, b) in self.lora_modules.items()},
                 "adapters": {name: {key: value.detach().cpu().clone()
                                      for key, value in module.state_dict().items()}
                              for name, module in self.adapters.items()}}
        torch.save(state, path)
        return str(path)

    def load_lora_only(self, load_path, device=None):
        """Restore weights into an identically configured, already injected model."""
        if not self.adapters:
            raise RuntimeError("Inject the model before loading plugin weights")
        device = torch.device(device) if device is not None else next(self.target_model.parameters()).device
        state = torch.load(load_path, map_location=device, weights_only=True)
        if state.get("metadata") != self._metadata():
            raise ValueError("Plugin metadata does not match this injector")
        if set(state["lora"]) != set(self.lora_modules) or set(state["adapters"]) != set(self.adapters):
            raise ValueError("Plugin module names do not match this model")
        for name, (a, b) in state["lora"].items():
            current_a, current_b = self.lora_modules[name]
            if a.shape != current_a.shape or b.shape != current_b.shape:
                raise ValueError(f"LoRA shape mismatch for {name}")
        for name, module in self.adapters.items():
            actual = state["adapters"][name]
            expected = module.state_dict()
            if set(actual) != set(expected) or any(actual[k].shape != v.shape for k, v in expected.items()):
                raise ValueError(f"Adapter shape mismatch for {name}")
        with torch.no_grad():
            for name, (a, b) in state["lora"].items():
                self.lora_modules[name][0].copy_(a)
                self.lora_modules[name][1].copy_(b)
            for name, module in self.adapters.items():
                module.load_state_dict(state["adapters"][name])
