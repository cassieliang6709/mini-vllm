import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_weights(model: nn.Module, weights):
    """Load an iterable of (hf_name, tensor) into `model`, sharding as each param's weight_loader says.

    A model may define `load_weight(name, tensor) -> bool` to claim weights that do not map 1:1
    onto a parameter (e.g. per-expert MoE weights stacked into one tensor).
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    custom = getattr(model, "load_weight", None)
    for weight_name, tensor in weights:
        if custom is not None and custom(weight_name, tensor):
            continue
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)


def _safetensors_iter(path: str):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_model(model: nn.Module, path: str):
    load_weights(model, _safetensors_iter(path))
