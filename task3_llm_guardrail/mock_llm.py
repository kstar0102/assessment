"""
Mock OpenAI style provider. Streams a canned reply in tiny random chunks so
PII lands across chunk boundaries.

Run:  uvicorn mock_llm:app --port 9100
"""

import asyncio
import json
import random
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM")

REPLY = (
    "Sure. The customer is reachable at john.doe+billing@example.co.uk and their "
    "SSN is 123-45-6789. Card on file: 4111 1111 1111 1111, backup 5500-0000-0000-0004. "
    "Order 1234567890123 is not a card. Thanks!"
)
CHUNK_DELAY = 0.01


def chunks(text: str, seed: int):
    rnd = random.Random(seed)
    i = 0
    while i < len(text):
        n = rnd.randint(1, 6)
        yield text[i : i + n]
        i += n


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    text = body.get("mock_reply", REPLY)
    seed = body.get("seed", 7)
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": "mock-1"}

    if not body.get("stream"):
        return JSONResponse({**base, "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    async def gen():
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        for part in chunks(text, seed):
            await asyncio.sleep(CHUNK_DELAY)
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
