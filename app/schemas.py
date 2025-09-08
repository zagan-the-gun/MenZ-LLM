from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class LlmRequest(BaseModel):
    type: str = "llm_request"
    payload: Dict[str, Any] = Field(default_factory=dict)