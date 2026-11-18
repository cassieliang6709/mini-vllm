"""Mixture-of-experts layer with expert parallelism (EP).

Per MoE layer, on every rank (each rank holds its OWN tokens; attention is data parallel):

  1. route      softmax(router logits) -> top-k experts + weights per token
  2. dispatch   a token goes to every rank that owns one of its k experts: sort the
                T*k (token, expert) assignments by destination rank, exchange counts,
                then all_to_all_single the hidden rows and the local expert ids
  3. compute    on the owning rank: sort received rows by expert, run grouped GEMMs
                (gate+up -> SiLU*mul -> down), un-sort
  4. combine    all_to_all_single back with the split sizes swapped; rows arrive in the order
                they were sent, so un-permute with the saved sort index, multiply by the routing
                weights and sum over k -> original token order and positions

Expert e lives on rank e // experts_per_rank. With ep_size == 1 steps 2 and 4 are skipped.

Grouped GEMM: out[i] = x[i] @ W[expert_of_row_i].T for rows sorted by expert.
  - GPU: Triton kernel (`grouped_gemm_triton`): rows are padded per expert to BLOCK_M so each
    BLOCK_M-row tile belongs to one expert (`align_rows`), one program per (row tile, N tile).
  - CPU / fallback: loop over experts with plain matmuls (`grouped_gemm_torch`).
The Triton kernel is not exercised on CPU. Its tile bookkeeping (`align_rows`) is tested with a
torch emulation of the kernel's indexing (`grouped_gemm_blocked_reference`).
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.utils.parallel import get_ep_rank, get_ep_size

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------- routing

def route(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int, norm_topk_prob: bool):
    """Returns (topk_weights [T,k] in x.dtype, topk_ids [T,k] int64). Softmax over ALL experts
    first, then top-k (OLMoE); weights are renormalised only if norm_topk_prob."""
    logits = F.linear(x, gate_weight)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    w, ids = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.to(x.dtype), ids


def load_balance_stats(topk_ids: torch.Tensor, num_experts: int, ep_size: int, my_rank: int | None = None) -> dict:
    """Load-balance numbers for one routing decision (this rank's tokens).

    expert_counts       tokens per expert
    rank_counts         assignments per destination rank
    imbalance_ratio     max/mean over experts (1.0 = perfect)
    rank_imbalance      max/mean over destination ranks
    cv                  std/mean over experts
    experts_unused      experts that got no token
    remote_fraction     share of assignments leaving this rank (needs my_rank)
    """
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).float()
    per_rank = counts.view(ep_size, -1).sum(1)
    mean = counts.mean().clamp_min(1e-9)
    out = dict(
        expert_counts=counts.long().tolist(),
        rank_counts=per_rank.long().tolist(),
        imbalance_ratio=float(counts.max() / mean),
        rank_imbalance=float(per_rank.max() / per_rank.mean().clamp_min(1e-9)),
        cv=float(counts.std(unbiased=False) / mean),
        experts_unused=int((counts == 0).sum()),
    )
    if my_rank is not None:
        total = per_rank.sum().clamp_min(1)
        out["remote_fraction"] = float((total - per_rank[my_rank]) / total)
    return out


# ---------------------------------------------------------------- grouped GEMM

def grouped_gemm_torch(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """x [M,K] rows sorted by expert, w [E,N,K], counts [E] (sum == M) -> [M,N]."""
    out = x.new_empty(x.size(0), w.size(1))
    start = 0
    for e, n in enumerate(counts.tolist()):
        if n:
            out[start:start + n] = x[start:start + n] @ w[e].T
        start += n
    return out


def align_rows(counts: torch.Tensor, block_m: int):
    """Pad each expert's row segment to a multiple of block_m so every block_m-row tile
    has exactly one expert.

    Returns
      row_ids       [num_blocks*block_m] int32: index into the sorted x, -1 for padding
      block_expert  [num_blocks] int32: expert of each tile
    """
    counts_l = counts.tolist()
    rows, experts, start = [], [], 0
    for e, n in enumerate(counts_l):
        nb = (n + block_m - 1) // block_m
        if nb:
            r = torch.full((nb * block_m,), -1, dtype=torch.int32)
            r[:n] = torch.arange(start, start + n, dtype=torch.int32)
            rows.append(r)
            experts += [e] * nb
        start += n
    if not rows:
        return (torch.empty(0, dtype=torch.int32, device=counts.device),) * 2
    return (torch.cat(rows).to(counts.device), torch.tensor(experts, dtype=torch.int32, device=counts.device))


def grouped_gemm_blocked_reference(x, w, counts, block_m=4):
    """Pure-torch emulation of exactly what the Triton kernel does per tile (row gather via
    row_ids, expert from block_expert, masked store). Used to test `align_rows`."""
    row_ids, block_expert = align_rows(counts, block_m)
    out = x.new_zeros(x.size(0), w.size(1))
    for b, e in enumerate(block_expert.tolist()):
        ids = row_ids[b * block_m:(b + 1) * block_m].long()
        ids = ids[ids >= 0]
        out[ids] = x[ids] @ w[e].T
    return out


if HAS_TRITON:
    @triton.jit
    def _grouped_gemm_kernel(
        a_ptr, w_ptr, c_ptr, row_ids_ptr, block_expert_ptr,
        N, K,
        stride_am, stride_ak, stride_we, stride_wn, stride_wk, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.load(block_expert_ptr + pid_m)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = tl.load(row_ids_ptr + offs_m)
        row_mask = rows >= 0
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
        w_ptrs = w_ptr + expert * stride_we + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        c_ptrs = c_ptr + rows[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


def grouped_gemm_triton(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor,
                        block_m=64, block_n=64, block_k=32) -> torch.Tensor:
    assert HAS_TRITON and x.is_cuda
    M, K = x.shape
    E, N, _ = w.shape
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    row_ids, block_expert = align_rows(counts.cpu(), block_m)
    if row_ids.numel() == 0:
        return out
    row_ids, block_expert = row_ids.to(x.device), block_expert.to(x.device)
    grid = (block_expert.numel(), triton.cdiv(N, block_n))
    _grouped_gemm_kernel[grid](x, w, out, row_ids, block_expert, N, K,
                               x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
                               out.stride(0), out.stride(1),
                               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return out


def grouped_gemm(x, w, counts):
    if HAS_TRITON and x.is_cuda:
        return grouped_gemm_triton(x, w, counts)
    return grouped_gemm_torch(x, w, counts)


# ---------------------------------------------------------------- experts + EP

class ExpertParallelMoE(nn.Module):
    """Top-k routed SwiGLU experts; holds only the experts this rank owns.

    w13: [E_local, 2*I, H] (gate rows first, then up)   w2: [E_local, H, I]
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool = False):
        super().__init__()
        self.ep_size, self.ep_rank = get_ep_size(), get_ep_rank()
        assert num_experts % self.ep_size == 0
        self.num_experts, self.top_k = num_experts, top_k
        self.experts_per_rank = num_experts // self.ep_size
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size))    # replicated router
        self.w13 = nn.Parameter(torch.empty(self.experts_per_rank, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.experts_per_rank, hidden_size, intermediate_size))
        self.act_fn = SiluAndMul()
        self.collect_stats = False
        self.last_stats: dict | None = None

    # ---- weight loading (called from the model's load_weight hook) ----
    def load_expert(self, expert_id: int, kind: str, tensor: torch.Tensor):
        """kind in gate|up|down; per-expert hub checkpoint layout. No-op for experts of other ranks."""
        local = expert_id - self.ep_rank * self.experts_per_rank
        if not 0 <= local < self.experts_per_rank:
            return
        I = self.intermediate_size
        if kind == "gate":
            self.w13.data[local, :I].copy_(tensor)
        elif kind == "up":
            self.w13.data[local, I:].copy_(tensor)
        else:
            self.w2.data[local].copy_(tensor)

    def load_fused(self, kind: str, tensor: torch.Tensor):
        """HF-v5 fused layout: gate_up_proj [E,2I,H] or down_proj [E,H,I]; keeps this rank's slice."""
        sl = slice(self.ep_rank * self.experts_per_rank, (self.ep_rank + 1) * self.experts_per_rank)
        (self.w13 if kind == "gate_up_proj" else self.w2).data.copy_(tensor[sl])

    # ---- compute on already-dispatched rows ----
    def _local_experts(self, x: torch.Tensor, local_ids: torch.Tensor) -> torch.Tensor:
        """x [R,H] rows, local_ids [R] in [0, E_local) -> [R,H] expert outputs (same row order)."""
        order = torch.argsort(local_ids, stable=True)
        counts = torch.bincount(local_ids, minlength=self.experts_per_rank)
        xs = x[order]
        h = self.act_fn(grouped_gemm(xs, self.w13, counts))
        ys = grouped_gemm(h, self.w2, counts)
        out = torch.empty_like(ys)
        out[order] = ys
        return out

    def dispatch(self, x, topk_ids):
        """Returns (recv_x, recv_local_ids, state) where state is what `combine` needs."""
        T, k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        dest = flat_ids // self.experts_per_rank
        order = torch.argsort(dest, stable=True)                  # assignments grouped by dest rank
        send_x = x[order // k]                                     # row of the owning token
        send_local = (flat_ids[order] % self.experts_per_rank).to(torch.int64)
        send_counts = torch.bincount(dest, minlength=self.ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        s, r = send_counts.tolist(), recv_counts.tolist()          # host sync: split sizes are data dependent
        recv_x = x.new_empty(sum(r), x.size(1))
        recv_local = torch.empty(sum(r), dtype=torch.int64, device=x.device)
        dist.all_to_all_single(recv_x, send_x, output_split_sizes=r, input_split_sizes=s)
        dist.all_to_all_single(recv_local, send_local, output_split_sizes=r, input_split_sizes=s)
        return recv_x, recv_local, (order, s, r, T, k)

    def combine(self, y, state, topk_weights):
        """y: expert outputs in the order rows were received. Returns [T,H] in original token order."""
        order, s, r, T, k = state
        back = torch.empty(sum(s), y.size(1), dtype=y.dtype, device=y.device)
        dist.all_to_all_single(back, y, output_split_sizes=s, input_split_sizes=r)
        flat = torch.empty_like(back)
        flat[order] = back                                          # undo the sort: row t*k+j = (token t, j-th expert)
        flat = flat.view(T, k, -1)
        return (flat.float() * topk_weights.float().unsqueeze(-1)).sum(1).to(y.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_w, topk_ids = route(x, self.gate_weight, self.top_k, self.norm_topk_prob)
        if self.collect_stats:
            self.last_stats = load_balance_stats(topk_ids, self.num_experts, self.ep_size, self.ep_rank)
        if self.ep_size == 1:
            T, k = topk_ids.shape
            y = self._local_experts(x.repeat_interleave(k, 0), topk_ids.reshape(-1))
            return (y.view(T, k, -1).float() * topk_w.float().unsqueeze(-1)).sum(1).to(x.dtype)
        recv_x, recv_local, state = self.dispatch(x, topk_ids)
        y = self._local_experts(recv_x, recv_local)
        return self.combine(y, state, topk_w)
