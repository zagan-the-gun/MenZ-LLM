from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Deque, Tuple
from collections import deque

import orjson
import websockets
from websockets.client import WebSocketClientProtocol

from .config import get_settings
from .llm.providers.local_llama import LocalLlamaProvider
from .schemas import LlmRequest


logger = logging.getLogger("wipe_client")


class WipeClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.host = settings.client_host
        self.port = settings.client_port
        self.path = settings.client_path
        self.reconnect_initial_ms = settings.reconnect_initial_ms
        self.reconnect_max_ms = settings.reconnect_max_ms

        self.url = f"ws://{self.host}:{self.port}{self.path}"

        # choose LLM provider: LocalLlama only (no fallback)
        try:
            self.llm = LocalLlamaProvider.from_env(settings)
        except Exception as ex:
            logger.error("Failed to initialize LocalLlama: %s", ex)
            raise

        self._stop_event = asyncio.Event()
        # バッファ設定: comment_context_lines を流用（N 行溜まったら送る）
        self._buffer_max: int = max(1, settings.llm_comment_context_lines or 1)
        self._include_speaker: bool = bool(settings.llm_comment_include_speaker)
        self._buffer: Deque[Tuple[Optional[str], str]] = deque(maxlen=self._buffer_max)

    async def start(self) -> None:
        backoff_ms = self.reconnect_initial_ms
        while not self._stop_event.is_set():
            try:
                logger.info("connecting to %s", self.url)
                async with websockets.connect(self.url, max_queue=64) as ws:
                    logger.info("connected")
                    backoff_ms = self.reconnect_initial_ms
                    # Send initial comment on connect (README 準拠)
                    try:
                        init_msg = {"type": "comment", "comment": "ワイプAI接続完了"}
                        await ws.send(orjson.dumps(init_msg).decode())
                    except Exception as ex:  # noqa: BLE001
                        logger.debug("failed to send initial subtitle: %s", ex)
                    await self._run_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as ex:  # noqa: BLE001
                logger.warning("connection error: %s", ex)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, self.reconnect_max_ms)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _run_loop(self, ws: WebSocketClientProtocol) -> None:
        async for message_text in ws:
            try:
                data = orjson.loads(message_text)
            except Exception:
                logger.debug("non-json message ignored")
                continue

            if not isinstance(data, dict):
                continue

            msg_type = data.get("type")
            if msg_type != "subtitle":
                # ignore other message types for now
                continue

            text = str(data.get("text") or data.get("payload", {}).get("text") or "")
            speaker = data.get("speaker") or data.get("payload", {}).get("speaker") or None
            if not text:
                continue
            logger.info("[DEBUG] recv subtitle: %s%s", text, f" (speaker={speaker})" if speaker else "")

            # バッファに積み、満杯になったらまとめて送る
            self._buffer.append((speaker, text))
            if len(self._buffer) < self._buffer_max:
                continue

            # 溜まった N 行の文脈を構築
            lines: list[str] = []
            for spk, sub in list(self._buffer):
                if self._include_speaker and spk:
                    lines.append(f"[{spk}] {sub}")
                else:
                    lines.append(sub)
            context_override = "\n".join(lines)
            last_speaker, last_text = self._buffer[-1]
            self._buffer.clear()

            req = LlmRequest(
                type="llm_request",
                payload={
                    "text": last_text,
                    **({"speaker": last_speaker} if last_speaker else {}),
                    "context_override": context_override,
                },
            )
            try:
                comment = await self.llm.generate_comment(req)
            except Exception as ex:  # noqa: BLE001
                logger.exception("llm error: %s", ex)
                comment = "(エラー)"

            if not (comment and comment.strip()):
                logger.info("[DEBUG] skip sending empty comment")
                continue
            reply: Dict[str, Any] = {
                "type": "comment",
                "comment": comment,
            }
            logger.info("[DEBUG] send comment: %s", comment)
            await ws.send(orjson.dumps(reply).decode())
