import torch
from nanovllm.utils.compile import maybe_compile
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
