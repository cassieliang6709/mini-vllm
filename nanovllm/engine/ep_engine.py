"""Expert-parallel engine for MoE models: one EPEngine per rank (GPU), data-parallel attention.

    rank r: own Scheduler + own KV cache + own requests  ->  dense layers run locally
            MoE layers: all_to_all dispatch -> experts on the owning rank -> all_to_all combine

The MoE all_to_all is a collective, so every rank must run the same number of model forwards in the
same order. When a rank has no work but another rank does, it runs a 1-token dummy forward
(KV writes are skipped: slot -1) so the collectives still line up. `step()` agrees on this with one
all_reduce of a "has work" flag. This engine needs a GPU + NCCL and has not been run yet (GPU-pending).

Launch one process per rank, e.g. with torch.multiprocessing.spawn (see bench/ep_bench.py).
"""
from time import perf_counter
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class EPEngine:

    def __init__(self, model: str, rank: int, expert_parallel_size: int, **kwargs):
        config = Config(model, expert_parallel_size=expert_parallel_size, **kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.rank = rank
        self.runner = ModelRunner(config, rank, [])
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.config = config
        self.scheduler = Scheduler(config)

    def add_request(self, prompt, sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        seq.token_times = []
        self.scheduler.add(seq)
        return seq

    def cache_stats(self) -> dict:
        return self.scheduler.block_manager.stats()

    def step(self) -> bool:
        """One lockstep iteration across all ranks. Returns False when every rank is idle."""
        flag = torch.tensor([0 if self.scheduler.is_finished() else 1], device="cuda")
        dist.all_reduce(flag)
        if flag.item() == 0:
            return False
        if self.scheduler.is_finished():
            dummy = Sequence([0])
            dummy.num_scheduled_tokens = 1
            self.runner.run([dummy], False)    # keeps the MoE all_to_all calls matched
            self.last_step_info = []
            return True
        seqs, is_pure_decode = self.scheduler.schedule()
        self.last_step_info = [(s.num_cached_tokens, s.num_scheduled_tokens, s.is_prefill) for s in seqs]
        token_ids = self.runner.run(seqs, is_pure_decode)
        emitted = self.scheduler.postprocess(seqs, token_ids)
        now = perf_counter()
        for seq, _ in emitted:
            seq.token_times.append(now)
        return True

    def generate(self, prompts, sampling_params):
        seqs = [self.add_request(p, sampling_params) for p in prompts]
        while self.step():
            pass
        return [s.completion_token_ids for s in seqs]
