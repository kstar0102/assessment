"""
Mock model providers with switchable failure modes.

  MOCK_MODE=ok|429|500|400|slow|garbage   (slow sleeps MOCK_SLOW_S, default 5s)

Run two for a local demo:
  MOCK_MODE=429 uvicorn mock_providers:app --port 9201   # primary, broken
  MOCK_MODE=ok  uvicorn mock_providers:app --port 9202   # secondary, healthy
"""

import asyncio
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse


def make_app(name: str = "mock", mode: str | None = None) -> FastAPI:
    app = FastAPI(title=f"Mock provider {name}")
    app.state.mode = mode or os.environ.get("MOCK_MODE", "ok")
    app.state.slow_s = float(os.environ.get("MOCK_SLOW_S", 5))
    app.state.hits = 0

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        app.state.hits += 1
        body = await request.json()
        mode = app.state.mode
        if mode == "429":
            return JSONResponse({"error": {"message": "quota exceeded for org-123 (internal shard 7)"}}, status_code=429)
        if mode == "500":
            return PlainTextResponse('Traceback (most recent call last):\n  File "/srv/llm/infer.py", line 88\nKeyError: weights', status_code=500)
        if mode == "400":
            return JSONResponse({"error": {"message": "max_tokens too large for model internal-v9"}}, status_code=400)
        if mode == "garbage":
            return PlainTextResponse("<html>oops</html>")
        if mode == "slow":
            await asyncio.sleep(app.state.slow_s)
        return {
            "id": f"cmpl-{name}",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"hello from {name}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    return app


app = make_app(os.environ.get("MOCK_NAME", "mock"))
