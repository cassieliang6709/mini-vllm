from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Two policies, switched by `config.enable_mixed_batch`:

    - legacy (upstream nano-vllm): a step is either all-prefill or all-decode;
      only the head of the waiting queue may be chunked.
    - mixed: every running request first gets its 1 decode token, then the
      leftover token budget is filled with prefill chunks from any number of
      waiting requests (each may be chunked).

    Both policies share one state rule: a request's phase is derived from
    num_cached_tokens vs num_tokens (see Sequence.is_prefill), so postprocess
    has no prefill/decode branch.
    """

    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_mixed_batch = getattr(config, "enable_mixed_batch", True)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # ---------- shared pieces ----------

    def _schedule_prefill(self, scheduled: list[Sequence], budget: int, chunk_any: bool) -> int:
        """Fill `budget` tokens from the waiting queue (FCFS). Returns tokens used."""
        used = 0
        i = 0
        while i < len(self.waiting) and len(scheduled) < self.max_num_seqs:
            remaining = budget - used
            if remaining <= 0:
                break
            seq = self.waiting[i]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break    # out of KV blocks; keep FCFS, don't skip ahead
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_uncomputed_tokens
            if remaining < num_tokens and scheduled and not chunk_any:
                break    # legacy: only the first scheduled seq may be chunked
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            used += seq.num_scheduled_tokens
            scheduled.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                del self.waiting[i]
                self.running.append(seq)
            else:
                i += 1
        return used

    def _schedule_decode(self, num_already: int) -> tuple[list[Sequence], bool]:
        """One token per running seq; preempt from the tail when out of blocks."""
        decodes = []
        preempted = False
        while self.running and num_already + len(decodes) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                decodes.append(seq)
        self.running.extendleft(reversed(decodes))
        return decodes, preempted

    # ---------- policies ----------

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (seqs, is_pure_decode)."""
        if self.enable_mixed_batch:
            return self._schedule_mixed()
        return self._schedule_legacy()

    def _schedule_legacy(self):
        scheduled = []
        self._schedule_prefill(scheduled, self.max_num_batched_tokens, chunk_any=False)
        if scheduled:
            return scheduled, False
        decodes, _ = self._schedule_decode(0)
        assert decodes
        return decodes, True

    def _schedule_mixed(self):
        # decode runs first, so every running seq here has exactly 1 uncomputed token
        scheduled, preempted = self._schedule_decode(0)
        budget = self.max_num_batched_tokens - len(scheduled)
        # after a preemption memory is tight; don't re-admit the victim in the same step
        if not preempted or not scheduled:
            self._schedule_prefill(scheduled, budget, chunk_any=True)
        assert scheduled
        is_pure_decode = all(not seq.is_prefill for seq in scheduled)
        return scheduled, is_pure_decode

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[Sequence, int]]:
        """Returns (seq, new_token) for every seq that emitted a token this step."""
        emitted = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue    # mid-prefill chunk: no token sampled
            seq.append_token(token_id)
            emitted.append((seq, token_id))
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return emitted
