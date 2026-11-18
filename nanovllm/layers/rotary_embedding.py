import math
from functools import lru_cache
import torch
from nanovllm.utils.compile import maybe_compile
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def llama3_scale_inv_freq(inv_freq: torch.Tensor, factor: float, low_freq_factor: float,
                          high_freq_factor: float, original_max_position_embeddings: int) -> torch.Tensor:
    """Llama-3.1 RoPE scaling (same math as HF `_compute_llama3_parameters`):
    long wavelengths are stretched by `factor`, short ones are kept, and the band in
    between is interpolated smoothly."""
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if scaling:
            cfg = dict(scaling)
            kind = cfg.get("rope_type", cfg.get("type"))
            assert kind == "llama3", f"unsupported rope scaling {kind}"
            inv_freq = llama3_scale_inv_freq(inv_freq, cfg["factor"], cfg["low_freq_factor"],
                                             cfg["high_freq_factor"], cfg["original_max_position_embeddings"])
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @maybe_compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    scaling: tuple | None = None,    # hashable: tuple(sorted(rope_scaling.items()))
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, scaling)
    return rotary_emb
