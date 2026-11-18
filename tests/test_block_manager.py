"""CPU tests for the KV block manager: refcounts, LRU eviction, prefix cache."""
import random

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4


def mk(tokens):
    Sequence.block_size = BS
    return Sequence(list(tokens), SamplingParams(max_tokens=4))


def prefill(bm, seq):
    """What the scheduler does: allocate, 'compute' the uncached tokens, register hashes."""
    n = bm.can_allocate(seq)
    assert n != -1
    bm.allocate(seq, n)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return n


def check(bm, live):
    refs = {}
    for seq in live:
        for b in seq.block_table:
            refs[b] = refs.get(b, 0) + 1
    for blk in bm.blocks:
        assert blk.ref_count == refs.get(blk.block_id, 0), f"ref mismatch on {blk.block_id}"
    assert set(refs) == bm.used_block_ids
    free, ev, used = set(bm.free_block_ids), set(bm.evictable), bm.used_block_ids
    assert len(free) == len(bm.free_block_ids) and not (free & ev) and not (free & used) and not (ev & used)
    assert len(free) + len(ev) + len(used) == len(bm.blocks)
    for b in ev:                        # evictable blocks are registered cache entries
        assert bm.hash_to_block_id[bm.blocks[b].hash] == b
    for h, b in bm.hash_to_block_id.items():
        assert bm.blocks[b].hash == h and b not in free


def test_prefix_hit_after_free():
    bm = BlockManager(8, BS)
    a = mk(range(10))                   # 3 blocks, 2 full
    assert prefill(bm, a) == 0
    bm.deallocate(a)
    assert len(bm.evictable) == 2       # two full blocks stay cached
    b = mk(list(range(8)) + [99, 98, 97])
    assert prefill(bm, b) == 2          # both full blocks hit
    assert b.num_cached_tokens == 11    # prefill() advanced it past the 8 hit tokens
    assert bm.hit_tokens == 8 and bm.evictions == 0
    check(bm, [b])


def test_shared_prefix_refcount():
    bm = BlockManager(8, BS)
    a = mk(range(9))
    prefill(bm, a)
    b = mk(list(range(8)) + [50])
    assert prefill(bm, b) == 2
    assert a.block_table[:2] == b.block_table[:2]
    assert all(bm.blocks[x].ref_count == 2 for x in a.block_table[:2])
    check(bm, [a, b])
    bm.deallocate(a)
    assert all(bm.blocks[x].ref_count == 1 for x in b.block_table[:2])
    check(bm, [b])
    bm.deallocate(b)
    check(bm, [])
    assert not bm.used_block_ids


def test_last_block_never_matched():
    bm = BlockManager(8, BS)
    a = mk(range(8))                    # exactly 2 full blocks
    prefill(bm, a)
    bm.deallocate(a)
    b = mk(range(8))
    assert bm.can_allocate(b) == 1      # block 1 is recomputed so >=1 token runs


def test_lru_order_and_eviction_under_pressure():
    bm = BlockManager(4, BS)
    seqs = [mk([100 * i + j for j in range(BS + 1)]) for i in range(2)]    # 2 blocks each, 1 full
    for s in seqs:
        prefill(bm, s)
    # release order: seqs[0] first (oldest), then seqs[1]
    bm.deallocate(seqs[0])
    bm.deallocate(seqs[1])
    first_full_blocks = [bm.hash_to_block_id[BlockManager.compute_hash(s.block(0))] for s in seqs]
    assert list(bm.evictable) == first_full_blocks     # oldest released first
    assert len(bm.free_block_ids) == 2
    # a 4-block request consumes 2 free blocks then evicts in LRU order
    big = mk(range(1000, 1000 + 4 * BS))
    prefill(bm, big)
    assert bm.evictions == 2
    assert not bm.evictable
    assert BlockManager.compute_hash(seqs[0].block(0)) not in bm.hash_to_block_id
    check(bm, [big])


def test_lru_victim_is_least_recently_released():
    bm = BlockManager(4, BS)
    a, b = mk(range(5)), mk(range(100, 105))    # 2 blocks each, 1 full
    prefill(bm, a); prefill(bm, b)
    bm.deallocate(a)                    # a released first
    bm.deallocate(b)                    # -> 2 free (partial tails) + 2 evictable
    ha = BlockManager.compute_hash(a.block(0))
    hb = BlockManager.compute_hash(b.block(0))
    c = mk(range(200, 200 + 3 * BS))    # needs 3 blocks: 2 free + 1 evicted (a's)
    prefill(bm, c)
    assert bm.evictions == 1
    assert ha not in bm.hash_to_block_id and hb in bm.hash_to_block_id
    check(bm, [c])


def test_reviving_a_block_refreshes_its_recency():
    bm = BlockManager(4, BS)
    a, b = mk(range(5)), mk(range(100, 105))
    prefill(bm, a); prefill(bm, b)
    bm.deallocate(a); bm.deallocate(b)            # LRU: a, b
    a2 = mk(list(range(4)) + [7])
    assert prefill(bm, a2) == 1                   # revive a's block
    bm.deallocate(a2)                             # now a is the most recent
    hb = BlockManager.compute_hash(b.block(0))
    ha = BlockManager.compute_hash(a.block(0))
    assert list(bm.evictable)[0] == bm.hash_to_block_id[hb]
    assert list(bm.evictable)[-1] == bm.hash_to_block_id[ha]


def test_can_allocate_rejects_when_full_and_counts_revived_blocks():
    bm = BlockManager(3, BS)
    a = mk(range(9))                    # 3 blocks, all used
    prefill(bm, a)
    assert bm.can_allocate(mk(range(50, 55))) == -1
    bm.deallocate(a)                    # 2 evictable + 1 free
    # same prefix, 3 blocks: 2 revived (leave pool) + 1 new from the free list -> fits exactly
    assert bm.can_allocate(mk(list(range(8)) + [1])) == 2
    # 4 blocks needed but only 3 exist
    assert bm.can_allocate(mk(range(13))) == -1


def test_can_append_and_may_append_use_evictable_blocks():
    bm = BlockManager(2, BS)
    a = mk(range(BS))                   # 1 full block
    prefill(bm, a)
    a.append_token(5)                   # len == BS+1 -> needs a new block
    assert bm.can_append(a)
    bm.may_append(a)
    assert len(a.block_table) == 2
    check(bm, [a])


def test_hash_collision_guard_checks_token_ids():
    bm = BlockManager(4, BS)
    a = mk(range(5))
    prefill(bm, a)
    bm.deallocate(a)
    h = BlockManager.compute_hash(a.block(0))
    bm.blocks[bm.hash_to_block_id[h]].token_ids = [9, 9, 9, 9]    # simulate a collision
    assert bm.can_allocate(mk(list(range(4)) + [1])) == 0


def test_duplicate_concurrent_prompts_do_not_leak():
    bm = BlockManager(8, BS)
    a, b = mk(range(9)), mk(range(9))
    # both allocate before either registers hashes (concurrent identical fresh prompts)
    for s in (a, b):
        bm.allocate(s, bm.can_allocate(s))
    for s in (a, b):
        s.num_scheduled_tokens = s.num_tokens
        bm.hash_blocks(s)
        s.num_cached_tokens = s.num_tokens
        s.num_scheduled_tokens = 0
    check(bm, [a, b])
    bm.deallocate(a); check(bm, [b])
    bm.deallocate(b); check(bm, [])
    assert len(bm.free_block_ids) + len(bm.evictable) == 8


@pytest.mark.parametrize("seed", range(5))
def test_randomized_workload_never_leaks_blocks(seed):
    rng = random.Random(seed)
    bm = BlockManager(24, BS)
    prefixes = [[rng.randint(0, 5) for _ in range(rng.randint(4, 12))] for _ in range(4)]
    live = []
    for _ in range(600):
        op = rng.random()
        if op < 0.45 or not live:
            toks = rng.choice(prefixes) + [rng.randint(0, 5) for _ in range(rng.randint(1, 10))]
            s = mk(toks)
            n = bm.can_allocate(s)
            if n != -1:
                bm.allocate(s, n)
                s.num_scheduled_tokens = s.num_tokens - s.num_cached_tokens
                bm.hash_blocks(s)
                s.num_cached_tokens += s.num_scheduled_tokens
                s.num_scheduled_tokens = 0
                live.append(s)
        elif op < 0.75:
            s = rng.choice(live)
            s.append_token(rng.randint(0, 5))          # scheduler order: len includes the pending token
            if not bm.can_append(s):
                s.token_ids.pop(); s.num_tokens -= 1; s.last_token = s.token_ids[-1]
            else:
                bm.may_append(s)
                # decode token computed: register a block if it just filled
                s.num_scheduled_tokens = 1
                s.num_cached_tokens = s.num_tokens - 1
                bm.hash_blocks(s)
                s.num_cached_tokens = s.num_tokens
                s.num_scheduled_tokens = 0
        else:
            s = live.pop(rng.randrange(len(live)))
            bm.deallocate(s)
        check(bm, live)
    for s in live:
        bm.deallocate(s)
    check(bm, [])
    assert not bm.used_block_ids
    assert bm.num_free_blocks == len(bm.blocks)
    st = bm.stats()
    assert st["hit_tokens"] <= st["queried_tokens"]
