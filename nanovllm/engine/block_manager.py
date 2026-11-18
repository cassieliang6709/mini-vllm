"""Paged KV-cache block manager with a prefix cache and LRU eviction.

Every physical block is in exactly one of three states:

    used       ref_count > 0. Owned by one or more sequences.
    evictable  ref_count == 0 and its content hash is registered in
               `hash_to_block_id`. The KV data is still valid, so a new request
               with the same prefix can revive it (prefix hit). Kept in
               `evictable`, an OrderedDict ordered oldest-released first.
    free       ref_count == 0, no registered content. Kept in `free_block_ids`.

Allocation takes from `free_block_ids` first. Only when it is empty does it
evict the least recently released block from `evictable` (and drops that block's
hash entry), so cached prefixes survive as long as there is memory to spare.

A block's recency is the moment its ref_count last dropped to 0. `deallocate`
walks a sequence's blocks tail-first, so the tail of a prompt is released
earlier than its head and is evicted first. A prefix hit has to match from block 0
onward (hashes are chained), so evicting a tail block never strands the head.

Prefix hashes chain: hash(block i) = H(hash(block i-1), tokens of block i).
`_lookup` also compares the stored token ids, so a 64-bit hash collision
cannot return wrong KV.
"""
from collections import OrderedDict, deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.evictable: OrderedDict[int, None] = OrderedDict()    # oldest released first
        self.used_block_ids: set[int] = set()
        # stats
        self.hit_tokens = 0          # prompt tokens served from the prefix cache
        self.queried_tokens = 0      # prompt tokens of all allocated sequences
        self.evictions = 0           # cached blocks dropped to make room

    # ---------- helpers ----------

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    @property
    def num_free_blocks(self) -> int:
        """Blocks that can be handed out: never-used/cleared plus evictable."""
        return len(self.free_block_ids) + len(self.evictable)

    def stats(self) -> dict:
        return dict(
            hit_tokens=self.hit_tokens,
            queried_tokens=self.queried_tokens,
            hit_rate=self.hit_tokens / self.queried_tokens if self.queried_tokens else 0.0,
            evictions=self.evictions,
            used_blocks=len(self.used_block_ids),
            cached_unreferenced_blocks=len(self.evictable),
            free_blocks=len(self.free_block_ids),
        )

    def _drop_hash(self, block: Block):
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    # ---------- block lifecycle ----------

    def _allocate_block(self) -> int:
        if self.free_block_ids:
            block_id = self.free_block_ids.popleft()
        else:
            block_id, _ = self.evictable.popitem(last=False)    # LRU victim
            self.evictions += 1
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self._drop_hash(block)
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _release_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_block_ids.remove(block_id)
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            self.evictable[block_id] = None        # newest end
        else:
            block.hash = -1
            block.token_ids = []
            self.free_block_ids.append(block_id)

    def _lookup(self, seq: Sequence) -> list[int]:
        """Block ids of the longest cached prefix. The last block of a sequence is
        never matched, so at least one token is always computed."""
        h = -1
        hits = []
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            hits.append(block_id)
        return hits

    # ---------- sequence-level API (used by the scheduler) ----------

    def can_allocate(self, seq: Sequence) -> int:
        """Number of cached prefix blocks if `seq` fits, else -1."""
        hits = self._lookup(seq)
        revived = sum(1 for b in hits if b in self.evictable)    # leave the evictable pool
        new_blocks = seq.num_blocks - len(hits)
        if self.num_free_blocks - revived < new_blocks:
            return -1
        return len(hits)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        hits = self._lookup(seq)[:num_cached_blocks]
        assert len(hits) == num_cached_blocks
        for block_id in hits:
            block = self.blocks[block_id]
            if block.ref_count > 0:
                block.ref_count += 1
            else:
                del self.evictable[block_id]
                block.ref_count = 1
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        self.hit_tokens += seq.num_cached_tokens
        self.queried_tokens += seq.num_prompt_tokens

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):    # tail first -> tail is evicted first
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return self.num_free_blocks >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """Register blocks that became full after this step's tokens are computed."""
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            # if an identical block is already registered (two requests with the same
            # fresh prompt ran concurrently), keep the first; this one just frees later.
            self.hash_to_block_id.setdefault(h, block.block_id)
