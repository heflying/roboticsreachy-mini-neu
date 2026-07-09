"""In-memory fake Qwen realtime WebSocket used by unit tests."""

from __future__ import annotations

import json
import asyncio
from typing import Any


class FakeQwenWebSocket:
    """Minimal websocket-like object for the Qwen handler tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        message = await self.incoming.get()
        return json.dumps(message)

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put({"type": "closed"})


class FakeQwenConnectContext:
    """Async context manager returned by a fake ``websockets.connect``."""

    def __init__(self, websocket: FakeQwenWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeQwenWebSocket:
        return self.websocket

    async def __aexit__(self, *_args: Any) -> bool:
        return False
