"""CPU test of the OpenAI server: real Scheduler, fake model + tokenizer."""
import json
from types import SimpleNamespace
from time import perf_counter

from fastapi.testclient import TestClient

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from server import AsyncEngine, build_app

EOS = 0


class FakeTok:
    eos_token_id = EOS
    def apply_chat_template(self, msgs, **_): return msgs[-1]["content"]
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids, **_): return "".join(chr(i) for i in ids)


class FakeLLM:
    """Echoes the prompt back one char per step, then EOS."""
    def __init__(self):
        Sequence.block_size = 4
        self.tokenizer = FakeTok()
        self.scheduler = Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=8, eos=EOS,
                                                   kvcache_block_size=4, num_kvcache_blocks=64))
    def add_request(self, ids, sp):
        seq = Sequence(ids, sp); seq.token_times = []; self.scheduler.add(seq); return seq
    def is_finished(self): return self.scheduler.is_finished()
    def step(self):
        seqs, _ = self.scheduler.schedule()
        out = []
        for s in seqs:
            k = s.num_completion_tokens
            out.append(s.prompt_token_ids[k] if k < s.num_prompt_tokens else EOS)
        self.last_emitted = self.scheduler.postprocess(seqs, out)


client = TestClient(build_app(AsyncEngine(FakeLLM()), "fake"))


def test_chat_non_stream():
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "héllo"}], "max_tokens": 50})
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "héllo"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 5


def test_chat_stream_deltas_concat_and_done():
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "abc"}], "stream": True}) as r:
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "abc"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_completion_length_limit():
    r = client.post("/v1/completions", json={"prompt": "abcdef", "max_tokens": 3})
    c = r.json()["choices"][0]
    assert c["text"] == "abc" and c["finish_reason"] == "length"
