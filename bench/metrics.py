"""MFU (prefill) and MBU (decode) for the benchmarks. Pure python, no torch.

Peak figures, RTX 4090 (AD102), NVIDIA Ada GPU Architecture whitepaper V2.02,
Appendix A "GeForce RTX 4090 Full Specifications":
  - Peak BF16 Tensor TFLOPS with FP32 accumulate: 165.2 dense (330.4 with 2:4 sparsity)
  - Peak FP16 Tensor TFLOPS with FP16 accumulate: 330.3 dense (660.6 sparse)
  - Memory bandwidth: 1008 GB/s
bf16 matmuls accumulate in fp32, and there is no bf16-with-bf16-accumulate mode, so the correct
dense peak for a bf16 model is 165.2 TFLOPS. 330.3/330.4 are different things:
FP16-with-FP16-accumulate dense, or BF16 with 2:4 sparsity. Using them would halve every MFU.
(https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-V2.02.pdf)

Formulas
--------
MFU (prefill) = achieved FLOPs/s / (peak FLOPs/s * num_gpus)
    FLOPs of a prefill chunk of q new tokens after c cached tokens:
        2 * N_active * q                                   (matmuls: 2 FLOPs per weight per token)
      + 4 * L * n_heads * head_dim * sum_{i=1..q} (c + i)  (QK^T and PV, causal)
    N_active = parameters touched per token, excluding the embedding table
    (a lookup, no FLOPs) but including lm_head. For MoE only the top-k experts count.
    The lm_head is applied to the last token of a sequence only; we count it for every
    token (a small overestimate, <1% for 8B).

MBU (decode)  = bytes read per step / step_time / (peak bandwidth * num_gpus)
    bytes per step = weight bytes read + KV bytes read
      weights: all non-expert params, plus for MoE the experts that were actually hit.
               With B tokens in the step and top-k of E experts, assuming uniform routing,
               expected distinct experts = E * (1 - (1 - k/E)^B).
      KV:      sum over sequences of context_len * 2 * L * n_kv_heads * head_dim * dtype_bytes.
    Writes of the new KV entry and activations are ignored (small).
Under TP/EP these totals are across all GPUs, so we divide by num_gpus * per-GPU peak.
Replicated parts (attention weights under DP-attention) are counted once per GPU that reads them;
pass `replicated_dense_copies` for that.
"""
from dataclasses import dataclass

PEAK = {
    # name: (dense bf16 tensor FLOP/s with fp32 accumulate, memory bytes/s)
    "NVIDIA GeForce RTX 4090": (165.2e12, 1008e9),
}


def peak_for(gpu_name: str):
    for k, v in PEAK.items():
        if k.lower() in gpu_name.lower():
            return v
    return None    # unknown GPU: MFU/MBU are reported as null, never guessed


@dataclass
class ModelShape:
    layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    dense_ffn: int = 0          # intermediate size for dense MLP (0 for MoE models)
    num_experts: int = 0
    top_k: int = 0
    expert_ffn: int = 0         # per-expert intermediate size
    dtype_bytes: int = 2
    qk_norm_full: bool = False

    @classmethod
    def from_hf(cls, c, dtype_bytes=2):
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        E = getattr(c, "num_experts", 0) or 0
        return cls(layers=c.num_hidden_layers, hidden=c.hidden_size, n_heads=c.num_attention_heads,
                   n_kv_heads=c.num_key_value_heads, head_dim=head_dim, vocab=c.vocab_size,
                   dense_ffn=0 if E else c.intermediate_size, num_experts=E,
                   top_k=getattr(c, "num_experts_per_tok", 0) or 0,
                   expert_ffn=c.intermediate_size if E else 0, dtype_bytes=dtype_bytes)

    # ---- parameter counts ----
    @property
    def attn_params_per_layer(self):
        q = self.hidden * self.n_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        return q + kv + o

    @property
    def expert_params(self):    # gate, up, down
        return 3 * self.hidden * self.expert_ffn

    @property
    def dense_params_per_layer(self):    # everything except experts
        p = self.attn_params_per_layer
        if self.num_experts:
            p += self.hidden * self.num_experts    # router
        else:
            p += 3 * self.hidden * self.dense_ffn
        return p

    @property
    def lm_head_params(self):
        return self.vocab * self.hidden

    def total_params(self):
        emb = self.vocab * self.hidden
        per_layer = self.dense_params_per_layer + self.num_experts * self.expert_params
        return emb + self.layers * per_layer + self.lm_head_params

    def active_params(self):
        """Params that cost FLOPs per token (no embedding lookup)."""
        per_layer = self.dense_params_per_layer + self.top_k * self.expert_params
        return self.layers * per_layer + self.lm_head_params


def prefill_flops(m: ModelShape, chunks):
    """chunks: iterable of (cached_tokens, new_tokens)."""
    total = 0
    for c, q in chunks:
        ctx_sum = q * c + q * (q + 1) // 2
        total += 2 * m.active_params() * q + 4 * m.layers * m.n_heads * m.head_dim * ctx_sum
    return total


def mfu(m: ModelShape, chunks, step_time_s, peak_flops, num_gpus=1):
    return prefill_flops(m, chunks) / step_time_s / (peak_flops * num_gpus)


def expected_distinct_experts(E, k, tokens):
    if E == 0:
        return 0
    return E * (1 - (1 - k / E) ** tokens)


def decode_bytes(m: ModelShape, context_lens, replicated_dense_copies=1):
    B = len(context_lens)
    dense = m.layers * m.dense_params_per_layer * replicated_dense_copies + m.lm_head_params
    experts = m.layers * expected_distinct_experts(m.num_experts, m.top_k, B) * m.expert_params
    weight_bytes = (dense + experts) * m.dtype_bytes
    kv_per_token = 2 * m.layers * m.n_kv_heads * m.head_dim * m.dtype_bytes
    return weight_bytes + sum(context_lens) * kv_per_token


def mbu(m: ModelShape, context_lens, step_time_s, peak_bw, num_gpus=1, replicated_dense_copies=1):
    return decode_bytes(m, context_lens, replicated_dense_copies) / step_time_s / (peak_bw * num_gpus)


class StepMetrics:
    """Accumulates per-step MFU (prefill-only steps) and MBU (pure-decode steps).
    Mixed steps are excluded from both: neither formula describes them cleanly."""

    def __init__(self, shape: ModelShape, gpu_name: str, num_gpus=1, replicated_dense_copies=1):
        self.m, self.n, self.rep = shape, num_gpus, replicated_dense_copies
        self.peak = peak_for(gpu_name)
        self.prefill_flops = self.prefill_time = 0.0
        self.decode_bytes = self.decode_time = 0.0

    def record(self, seqs_info, dt):
        """seqs_info: list of (cached_tokens_before, scheduled_tokens, is_prefill)."""
        pre = [x for x in seqs_info if x[2]]
        if pre and len(pre) == len(seqs_info):
            self.prefill_flops += prefill_flops(self.m, [(c, q) for c, q, _ in pre])
            self.prefill_time += dt
        elif not pre:
            self.decode_bytes += decode_bytes(self.m, [c + q for c, q, _ in seqs_info], self.rep)
            self.decode_time += dt

    def summary(self):
        if self.peak is None:
            return dict(mfu_prefill=None, mbu_decode=None, note="peak unknown for this GPU")
        pf, bw = self.peak
        return dict(
            mfu_prefill=round(self.prefill_flops / self.prefill_time / (pf * self.n), 4) if self.prefill_time else None,
            mbu_decode=round(self.decode_bytes / self.decode_time / (bw * self.n), 4) if self.decode_time else None,
        )
