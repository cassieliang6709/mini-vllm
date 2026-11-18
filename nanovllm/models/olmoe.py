"""OLMoE-1B-7B (allenai/OLMoE-1B-7B-0924): 16 layers, 64 experts, top-8, ~7B params (1B active).

Parallelism: attention / norms / embedding / lm_head are replicated and every rank runs them on its
own requests (data parallel); MoE experts are split across ranks (expert parallel). Build with
`set_parallel_mode("ep")` so the dense layers see tp_size == 1.

Differences from Llama: q_norm / k_norm are RMSNorms over the whole q / k projection (before the
head reshape), the MLP is a sparse MoE block, softmax-then-top-k routing without renormalisation
(norm_topk_prob = False in the released config).
"""
import re
import torch
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.moe import ExpertParallelMoE
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.llama import rope_settings
from nanovllm.utils.parallel import get_tp_size


class OlmoeAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert get_tp_size() == 1, "OLMoE q/k norm spans all heads: attention must be replicated (ep mode)"
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.clip_qkv = getattr(config, "clip_qkv", None)
        bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads, bias=bias)
        self.o_proj = RowParallelLinear(self.q_size, config.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.q_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.kv_size, eps=config.rms_norm_eps)
        theta, scaling = rope_settings(config)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
                                   max_position=config.max_position_embeddings, base=theta, scaling=scaling)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.clip_qkv is not None:
            q, k, v = (t.clamp(-self.clip_qkv, self.clip_qkv) for t in (q, k, v))
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))


class OlmoeSparseMoeBlock(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.experts = ExpertParallelMoE(config.hidden_size, config.intermediate_size, config.num_experts,
                                         config.num_experts_per_tok, getattr(config, "norm_topk_prob", False))

    def forward(self, x):
        return self.experts(x)


class OlmoeDecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = OlmoeAttention(config)
        self.mlp = OlmoeSparseMoeBlock(config)
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


class OlmoeModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert getattr(config, "decoder_sparse_step", 1) == 1 and not getattr(config, "mlp_only_layers", [])
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(?:(\d+)\.(gate|up|down)_proj\.weight|(gate_up_proj|down_proj))$")


class OlmoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = OlmoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def load_weight(self, name: str, tensor: torch.Tensor) -> bool:
        """Claim expert and router weights. Supports the hub layout (one tensor per expert) and the
        fused 3D layout of recent transformers."""
        m = _EXPERT_RE.match(name)
        if m:
            moe = self.model.layers[int(m.group(1))].mlp.experts
            if m.group(2) is not None:
                moe.load_expert(int(m.group(2)), m.group(3), tensor)
            else:
                moe.load_fused(m.group(4), tensor)
            return True
        if name.endswith(".mlp.gate.weight"):
            layer = int(name.split(".")[2])
            self.model.layers[layer].mlp.experts.gate_weight.data.copy_(tensor)
            return True
        return False

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
