from __future__ import annotations

from abc import ABC, abstractmethod

from ...schemas import LlmRequest


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def generate_comment(self, request: LlmRequest) -> str:  # pragma: no cover - interface
        raise NotImplementedError