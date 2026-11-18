"""Llama (3.x): GQA, RoPE with llama3 scaling, SwiGLU MLP, untied lm_head (8B).

Sharding (tensor parallel over `get_tp_size()` ranks, same scheme as qwen3.py):
  qkv_proj  column parallel (heads are split; n_heads and n_kv_heads must divide tp_size)
  o_proj    row parallel + all-reduce
  gate_up   column parallel, down_proj row parallel + all-reduce
  embedding / lm_head split over the vocab
"""
import torch
from torch import nn
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.parallel import get_tp_size


def rope_settings(config) -> tuple[float, tuple | None]:
    """(theta, hashable scaling or None) from either transformers>=5 `rope_parameters`
    or the older `rope_scaling` + `rope_theta`."""
    rp = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    theta = rp.get("rope_theta") or getattr(config, "rope_theta", None) or 10000.0
    kind = rp.get("rope_type", rp.get("type"))
    scaling = tuple(sorted(rp.items())) if kind not in (None, "default") else None
    return float(theta), scaling


class LlamaAttention(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        tp_size = get_tp_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0 and self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, self.total_num_heads,
                                          self.total_num_kv_heads, bias=bias)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=bias)
        theta, scaling = rope_settings(config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
                                   max_position=config.max_position_embeddings, base=theta, scaling=scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class LlamaMLP(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(config.hidden_size, [config.intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
