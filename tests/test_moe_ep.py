"""MoE + expert parallel on CPU (gloo, 2 processes). Needs torch (+ transformers for the HF check)."""
import _dist_helpers  # noqa: F401  (stub must come first)
import pytest

torch = pytest.importorskip("torch")

E, K, H, I = 8, 2, 16, 12


def full_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    return dict(gate=r(E, H), w13=r(E, 2 * I, H), w2=r(E, H, I))


def tokens_for(rank, n):
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randn(n, H, generator=g)


TOKENS = {0: 13, 1: 9}    # different counts per rank on purpose


def moe_reference(x, wts, norm_topk=False):
    """Plain per-token loop with the FULL weights; independent of the module under test."""
    probs = torch.softmax(x @ wts["gate"].T, -1)
    w, ids = probs.topk(K, -1)
    if norm_topk:
        w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for t in range(x.size(0)):
        for j in range(K):
            e = int(ids[t, j])
            gate, up = (wts["w13"][e] @ x[t]).chunk(2)
            out[t] += w[t, j] * (wts["w2"][e] @ (torch.nn.functional.silu(gate) * up))
    return out


def build_layer(wts, norm_topk=False):
    from nanovllm.layers.moe import ExpertParallelMoE
    layer = ExpertParallelMoE(H, I, E, K, norm_topk)
    layer.gate_weight.data.copy_(wts["gate"])
    for e in range(E):
        layer.load_expert(e, "gate", wts["w13"][e, :I])
        layer.load_expert(e, "up", wts["w13"][e, I:])
        layer.load_expert(e, "down", wts["w2"][e])
    return layer


# ---------------------------------------------------------------- EP output == single rank

def _layer_worker(rank, world, out_dir):
    from nanovllm.utils.parallel import set_parallel_mode
    set_parallel_mode("ep")
    layer = build_layer(full_weights())
    layer.collect_stats = True
    with torch.inference_mode():
        y = layer(tokens_for(rank, TOKENS[rank]))
    torch.save(dict(y=y, stats=layer.last_stats), f"{out_dir}/r{world}_{rank}.pt")


@pytest.mark.parametrize("world", [1, 2])
def test_ep_layer_matches_reference(world, tmp_path):
    _dist_helpers.run_ranks(_layer_worker, world, str(tmp_path))
    wts = full_weights()
    for rank in range(world):
        got = torch.load(tmp_path / f"r{world}_{rank}.pt")["y"]
        ref = moe_reference(tokens_for(rank, TOKENS[rank]), wts)
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (got - ref).abs().max()


def test_ep2_equals_ep1_same_tokens(tmp_path):
    """Rank 0's tokens give the same output whether experts are on 1 rank or split over 2."""
    _dist_helpers.run_ranks(_layer_worker, 1, str(tmp_path))
    _dist_helpers.run_ranks(_layer_worker, 2, str(tmp_path))
    a = torch.load(tmp_path / "r1_0.pt")["y"]
    b = torch.load(tmp_path / "r2_0.pt")["y"]
    assert torch.allclose(a, b, atol=1e-6)


def test_norm_topk_prob_flag_changes_weights():
    wts = full_weights()
    x = tokens_for(0, 5)
    assert not torch.allclose(moe_reference(x, wts, False), moe_reference(x, wts, True))


# ---------------------------------------------------------------- dispatch / combine round trip

def _roundtrip_worker(rank, world, out_dir):
    from nanovllm.utils.parallel import set_parallel_mode
    set_parallel_mode("ep")
    layer = build_layer(full_weights())
    n = TOKENS[rank]
    x = tokens_for(rank, n)
    g = torch.Generator().manual_seed(7 + rank)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])    # distinct experts per token
    w = torch.rand(n, K, generator=g)
    recv_x, recv_local, state = layer.dispatch(x, ids)
    # every received row is the hidden state of some token routed to one of OUR experts
    assert recv_local.min() >= 0 and recv_local.max() < layer.experts_per_rank
    glob = recv_local + rank * layer.experts_per_rank
    y = recv_x * (1 + glob.float()).unsqueeze(1)           # marks which expert processed the row
    out = layer.combine(y, state, w)
    expected = x * (w * (1 + ids.float())).sum(1, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max()
    torch.save(dict(recv=recv_x.shape[0]), f"{out_dir}/rt_{rank}.pt")


def test_dispatch_combine_roundtrip_preserves_order_and_positions(tmp_path):
    _dist_helpers.run_ranks(_roundtrip_worker, 2, str(tmp_path))
    # conservation: rows sent == rows received across ranks
    recv = sum(torch.load(tmp_path / f"rt_{r}.pt")["recv"] for r in range(2))
    assert recv == K * sum(TOKENS.values())


# ---------------------------------------------------------------- grouped GEMM + stats

@pytest.mark.parametrize("block_m", [1, 4, 8])
def test_blocked_grouped_gemm_matches_fallback(block_m):
    from nanovllm.layers.moe import grouped_gemm_torch, grouped_gemm_blocked_reference, align_rows
    g = torch.Generator().manual_seed(3)
    counts = torch.tensor([0, 5, 1, 0, 9, 4])
    M = int(counts.sum())
    x, w = torch.randn(M, 6, generator=g), torch.randn(6, 7, 6, generator=g)
    ref = torch.cat([x[s:s + n] @ w[e].T for e, (s, n) in enumerate(zip(torch.cumsum(counts, 0) - counts, counts.tolist())) if n])
    assert torch.allclose(grouped_gemm_torch(x, w, counts), ref, atol=1e-5)
    assert torch.allclose(grouped_gemm_blocked_reference(x, w, counts, block_m), ref, atol=1e-5)
    row_ids, block_expert = align_rows(counts, block_m)
    valid = row_ids[row_ids >= 0]
    assert sorted(valid.tolist()) == list(range(M))          # every row exactly once
    assert len(row_ids) == len(block_expert) * block_m
    for b, e in enumerate(block_expert.tolist()):            # tile -> single expert owning those rows
        for r in row_ids[b * block_m:(b + 1) * block_m].tolist():
            if r >= 0:
                start = int(torch.cumsum(counts, 0)[e] - counts[e])
                assert start <= r < start + int(counts[e])


def test_load_balance_stats():
    from nanovllm.layers.moe import load_balance_stats
    ids = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 5]])     # 8 experts, 2 ranks (0-3 | 4-7)
    s = load_balance_stats(ids, 8, 2, my_rank=0)
    assert s["expert_counts"] == [4, 2, 1, 0, 0, 1, 0, 0]
    assert s["rank_counts"] == [7, 1]
    assert s["experts_unused"] == 4
    assert s["imbalance_ratio"] == pytest.approx(4 / 1.0)
    assert s["rank_imbalance"] == pytest.approx(7 / 4)
    assert s["remote_fraction"] == pytest.approx(1 / 8)
    even = load_balance_stats(torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]]), 8, 2, my_rank=1)
    assert even["imbalance_ratio"] == 1.0 and even["rank_imbalance"] == 1.0 and even["cv"] == 0.0
    assert even["remote_fraction"] == 0.5


def test_layer_collects_stats(tmp_path):
    _dist_helpers.run_ranks(_layer_worker, 2, str(tmp_path))
    st = torch.load(tmp_path / "r2_0.pt")["stats"]
    assert sum(st["expert_counts"]) == K * TOKENS[0]
    assert len(st["rank_counts"]) == 2


# ---------------------------------------------------------------- full OLMoE model, EP vs 1 rank vs HF

SETS = {0: [5, 7], 1: [6]}


def tiny_olmoe_config():
    from transformers import OlmoeConfig
    return OlmoeConfig(vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=8, num_key_value_heads=8, num_experts=8, num_experts_per_tok=2,
                       max_position_embeddings=256, rms_norm_eps=1e-5, tie_word_embeddings=False,
                       norm_topk_prob=False, clip_qkv=None, rope_parameters={"rope_type": "default", "rope_theta": 10000.0})


def seqs_for(i):
    g = torch.Generator().manual_seed(11 + i)
    return [torch.randint(0, 128, (n,), generator=g) for n in SETS[i]]


def _run_olmoe_set(model, i):
    from nanovllm.utils.context import set_context, reset_context
    lens = SETS[i]
    ids = torch.cat(seqs_for(i))
    pos = torch.cat([torch.arange(n) for n in lens])
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32)
    with torch.inference_mode():
        set_context(True, cu, cu, max(lens), max(lens))
        h = model(ids, pos)
        set_context(False)
        logits = model.compute_logits(h)
    reset_context()
    return logits


def _olmoe_worker(rank, world, state_path, out_dir):
    from nanovllm.utils.parallel import set_parallel_mode
    from nanovllm.utils.loader import load_weights
    from nanovllm.models.olmoe import OlmoeForCausalLM
    set_parallel_mode("ep")
    model = OlmoeForCausalLM(tiny_olmoe_config())
    load_weights(model, torch.load(state_path).items())
    for i in (SETS if world == 1 else [rank]):    # EP: each rank serves its own requests (DP attention)
        torch.save(_run_olmoe_set(model, i), f"{out_dir}/w{world}_set{i}.pt")


@pytest.fixture(scope="module")
def olmoe_hf(tmp_path_factory):
    pytest.importorskip("transformers")
    from transformers import OlmoeForCausalLM as HF
    torch.manual_seed(0)
    hf = HF(tiny_olmoe_config()).eval().float()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
            elif "gate.weight" in n:
                p.mul_(8)    # peaky router so top-k choices are stable and varied
    path = tmp_path_factory.mktemp("o") / "state.pt"
    torch.save(hf.state_dict(), path)
    return hf, str(path)


def test_olmoe_ep_matches_single_rank_and_hf(olmoe_hf, tmp_path):
    hf, state = olmoe_hf
    _dist_helpers.run_ranks(_olmoe_worker, 1, state, str(tmp_path))
    _dist_helpers.run_ranks(_olmoe_worker, 2, state, str(tmp_path))
    for i in SETS:
        with torch.no_grad():
            ref = torch.cat([hf(s[None]).logits[0] for s in seqs_for(i)])
        one = torch.load(tmp_path / f"w1_set{i}.pt")
        two = torch.load(tmp_path / f"w2_set{i}.pt")
        assert torch.allclose(one, ref, atol=2e-4, rtol=1e-4), ("ep1 vs hf", i, (one - ref).abs().max())
        assert torch.allclose(two, one, atol=2e-5, rtol=1e-5), ("ep2 vs ep1", i, (two - one).abs().max())
