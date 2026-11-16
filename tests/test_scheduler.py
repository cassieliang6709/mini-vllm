"""CPU-only tests: scheduler + block manager, no torch / GPU needed.

A fake model samples token = 1000 + seq_id, so every step's outputs are
deterministic and we can check KV-cache bookkeeping after each step.
"""
import random
from types import SimpleNamespace

import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams

BS = 4    # tiny block size so block boundaries are hit often
EOS = -1


def make_sched(mixed=True, blocks=64, budget=16, max_seqs=8):
    Sequence.block_size = BS
    cfg = SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=budget, eos=EOS,
                          kvcache_block_size=BS, num_kvcache_blocks=blocks, enable_mixed_batch=mixed)
    return Scheduler(cfg)


def add(s, prompt_len, max_tokens=5, base=0):
    seq = Sequence([base + i for i in range(prompt_len)], SamplingParams(max_tokens=max_tokens))
    s.add(seq)
    return seq


def step(s):
    seqs, pure = s.schedule()
    snapshot = [(seq, seq.num_scheduled_tokens, seq.is_prefill) for seq in seqs]
    assert sum(n for _, n, _ in snapshot) <= s.max_num_batched_tokens
    assert len(seqs) <= s.max_num_seqs
    s.postprocess(seqs, [1000 + seq.seq_id for seq in seqs])
    check_invariants(s)
    return snapshot, pure


def check_invariants(s):
    bm = s.block_manager
    live = list(s.running) + [q for q in s.waiting if q.block_table]
    refs = {}
    for seq in live:
        assert len(seq.block_table) >= (seq.num_cached_tokens + BS - 1) // BS
        assert seq.num_cached_tokens <= seq.num_tokens
        for b in seq.block_table:
            refs[b] = refs.get(b, 0) + 1
    for b, n in refs.items():
        assert bm.blocks[b].ref_count == n, f"block {b}: ref {bm.blocks[b].ref_count} != owners {n}"
    assert set(refs) == bm.used_block_ids
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == len(bm.blocks)
    for seq in s.running:
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_uncomputed_tokens == 1    # running = ready for one decode token


def run_all(s, limit=10_000):
    for _ in range(limit):
        if s.is_finished():
            return
        step(s)
    raise AssertionError("did not finish")


# ---------- policy behavior ----------

def test_mixed_step_contains_decode_and_prefill_of_several_requests():
    s = make_sched(budget=10)
    a = add(s, 3)
    step(s)                                   # a fully prefilled -> running
    b, c = add(s, 6, base=100), add(s, 6, base=200)
    snap, pure = step(s)
    kinds = {seq.seq_id: (n, pre) for seq, n, pre in snap}
    assert kinds[a.seq_id] == (1, False)      # a decodes
    assert kinds[b.seq_id] == (6, True)       # b prefills whole prompt
    assert kinds[c.seq_id] == (3, True)       # c gets the leftover chunk (not queue head)
    assert not pure


def test_legacy_never_mixes_and_only_chunks_head():
    s = make_sched(mixed=False, budget=10)
    a = add(s, 3)
    step(s)
    add(s, 6, base=100), add(s, 6, base=200)
    snap, _ = step(s)
    assert all(pre for _, _, pre in snap)     # prefill-only step
    assert a not in [seq for seq, _, _ in snap]
    assert [n for _, n, _ in snap] == [6]     # second request is not chunked alongside


def test_mixed_decode_is_not_starved_by_long_prompt():
    s = make_sched(budget=8)
    a = add(s, 2, max_tokens=20)
    step(s)
    add(s, 64, base=500)                      # 8 steps of prefill
    for _ in range(5):
        snap, _ = step(s)
        assert any(seq is a and n == 1 for seq, n, _ in snap)


def test_chunked_prefill_emits_token_only_after_last_chunk():
    s = make_sched(budget=5)
    a = add(s, 12)
    for expected_cached in (5, 10):
        step(s)
        assert a.num_cached_tokens == expected_cached and a.num_completion_tokens == 0
    step(s)
    assert a.num_completion_tokens == 1 and a.status == SequenceStatus.RUNNING


@pytest.mark.parametrize("mixed", [True, False])
def test_outputs_identical_across_policies(mixed):
    """Same prompts -> each request emits exactly max_tokens deterministic tokens."""
    s = make_sched(mixed=mixed, budget=7)
    seqs = [add(s, n, max_tokens=m, base=k * 100) for k, (n, m) in enumerate([(5, 3), (9, 4), (2, 6), (13, 2)])]
    run_all(s)
    for seq in seqs:
        assert seq.completion_token_ids == [1000 + seq.seq_id] * seq.max_tokens
    assert not s.block_manager.used_block_ids


# ---------- cache state ----------

def test_prefix_cache_reuse_skips_cached_blocks():
    s = make_sched(budget=64)
    shared = list(range(12))                  # 3 full blocks
    a = Sequence(shared + [50, 51], SamplingParams(max_tokens=3)); s.add(a)
    step(s)
    b = Sequence(shared + [60, 61], SamplingParams(max_tokens=3)); s.add(b)
    snap, _ = step(s)
    n_b = next(n for seq, n, _ in snap if seq is b)
    assert n_b == 2                           # 12 shared tokens served from cache
    assert a.block_table[:3] == b.block_table[:3]
    run_all(s)
    assert not s.block_manager.used_block_ids


def test_preemption_under_memory_pressure_recovers():
    s = make_sched(blocks=6, budget=32, max_seqs=4)
    seqs = [add(s, 7, max_tokens=8, base=k * 100) for k in range(3)]
    run_all(s)
    for seq in seqs:
        assert seq.num_completion_tokens == 8
    assert not s.block_manager.used_block_ids


@pytest.mark.parametrize("mixed", [True, False])
def test_random_workload_invariants(mixed):
    rng = random.Random(0)
    s = make_sched(mixed=mixed, blocks=24, budget=rng.choice([6, 11, 16]), max_seqs=5)
    seqs = []
    for t in range(300):
        if rng.random() < 0.3 and len(seqs) < 40:
            seqs.append(add(s, rng.randint(1, 20), max_tokens=rng.randint(1, 12), base=rng.randint(0, 3) * 1000))
        if not s.is_finished():
            step(s)
    run_all(s)
    for seq in seqs:
        assert seq.is_finished and seq.num_completion_tokens == seq.max_tokens
    assert not s.block_manager.used_block_ids
