"""OpenAI-compatible server: POST /v1/chat/completions and /v1/completions (stream or not).

    python server.py --model ~/huggingface/Qwen3-0.6B --port 8000
    curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

One background thread owns the engine and calls step() in a loop; each HTTP
request registers an asyncio.Queue and receives its tokens as they are emitted.
New requests join the running batch on the next step (continuous batching).
"""
import argparse
import asyncio
import json
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanovllm.sampling_params import SamplingParams


class AsyncEngine:

    def __init__(self, llm):
        self.llm = llm
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.queues: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt_ids: list[int], sp: SamplingParams) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        with self.lock:
            seq = self.llm.add_request(prompt_ids, sp)
            self.queues[seq.seq_id] = (loop, q)
        self.wakeup.set()
        return q

    def _loop(self):
        while True:
            self.wakeup.wait()
            with self.lock:
                if self.llm.is_finished():
                    self.wakeup.clear()
                    continue
                self.llm.step()
                emitted = self.llm.last_emitted
            for seq, token_id in emitted:
                loop, q = self.queues[seq.seq_id]
                done = seq.is_finished
                if done:
                    del self.queues[seq.seq_id]
                    reason = "length" if seq.num_completion_tokens == seq.max_tokens else "stop"
                else:
                    reason = None
                loop.call_soon_threadsafe(q.put_nowait, (token_id, reason))


def build_app(engine: AsyncEngine, model_name: str) -> FastAPI:
    app = FastAPI()
    tok = engine.llm.tokenizer

    async def handle(body: dict, chat: bool):
        if chat:
            prompt = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True)
        else:
            prompt = body["prompt"]
        sp = SamplingParams(temperature=max(float(body.get("temperature", 1.0)), 1e-5),
                            max_tokens=int(body.get("max_tokens", 256)))
        prompt_ids = tok.encode(prompt)
        q = engine.submit(prompt_ids, sp)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        def chunk(text, reason):
            choice = {"index": 0, "finish_reason": reason}
            if chat:
                choice["delta"] = {"content": text} if text else {}
            else:
                choice["text"] = text
            return {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created,
                    "model": model_name, "choices": [choice]}

        if body.get("stream"):
            async def gen():
                ids, sent = [], ""
                while True:
                    token_id, reason = await q.get()
                    if token_id != tok.eos_token_id:
                        ids.append(token_id)
                    # decode the full prefix so multi-byte characters are not split
                    text = tok.decode(ids, skip_special_tokens=True)
                    delta, sent = text[len(sent):], text
                    if delta.endswith("�"):
                        sent = sent[:-1]; delta = delta[:-1]
                    if delta or reason:
                        yield f"data: {json.dumps(chunk(delta, reason), ensure_ascii=False)}\n\n"
                    if reason:
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        ids, reason = [], None
        while reason is None:
            token_id, reason = await q.get()
            if token_id != tok.eos_token_id:
                ids.append(token_id)
        text = tok.decode(ids, skip_special_tokens=True)
        choice = {"index": 0, "finish_reason": reason}
        choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
        return JSONResponse({"id": rid, "object": obj, "created": created, "model": model_name,
                             "choices": [choice],
                             "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(ids),
                                       "total_tokens": len(prompt_ids) + len(ids)}})

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        return await handle(await req.json(), chat=True)

    @app.post("/v1/completions")
    async def completions(req: Request):
        return await handle(await req.json(), chat=False)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--legacy-scheduler", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()
    import os
    from nanovllm import LLM
    path = os.path.expanduser(args.model)
    llm = LLM(path, max_model_len=args.max_model_len, enable_mixed_batch=not args.legacy_scheduler,
              enforce_eager=args.enforce_eager)
    app = build_app(AsyncEngine(llm), os.path.basename(path.rstrip("/")))
    uvicorn.run(app, host="0.0.0.0", port=args.port)
