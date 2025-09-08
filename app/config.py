from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import configparser


@dataclass(frozen=True)
class Settings:
    # LLM (llama.cpp) settings
    llm_model_name: str | None
    llm_quantization: str
    llm_n_ctx: int
    llm_n_threads: int
    llm_n_gpu_layers: int
    llm_temperature: float
    llm_max_tokens: int
    # WebSocket client (zagaroid) settings
    client_host: str
    client_port: int
    client_path: str
    reconnect_initial_ms: int
    reconnect_max_ms: int
    # Prompt settings
    llm_comment_prompt: str | None
    llm_comment_max_chars: int
    llm_comment_context_lines: int
    llm_comment_include_speaker: bool
    llm_notify_on_filtered_empty: bool


def _default_config_path() -> Path:
    # project root assumed as parent of this file's directory
    return (Path(__file__).resolve().parent.parent / "config.ini").resolve()


def _load_from_ini(path: Path) -> Settings:
    parser = configparser.ConfigParser()
    parser.read(path)

    llm = parser["llm"] if parser.has_section("llm") else {}
    client = parser["client"] if parser.has_section("client") else {}

    # llm (local llama.cpp)
    llm_model_name = llm.get("model_name") or None
    llm_quantization = llm.get("quantization", "Q4_K_M")
    llm_n_ctx = int(llm.get("n_ctx", "2048"))
    llm_n_threads = int(llm.get("n_threads", "0"))
    llm_n_gpu_layers = int(llm.get("n_gpu_layers", "0"))
    llm_temperature = float(llm.get("temperature", "0.3"))
    llm_max_tokens = int(llm.get("max_tokens", "10"))
    llm_comment_prompt = llm.get("comment_prompt") or None
    llm_comment_max_chars = int(llm.get("comment_max_chars", "40"))
    llm_comment_context_lines = int(llm.get("comment_context_lines", "0"))
    include_speaker_raw = llm.get("comment_include_speaker", "true").strip().lower()
    llm_comment_include_speaker = include_speaker_raw in ("1", "true", "yes", "on")
    notify_empty_raw = llm.get("notify_on_filtered_empty", "false").strip().lower()
    llm_notify_on_filtered_empty = notify_empty_raw in ("1", "true", "yes", "on")

    # client
    client_host = client.get("host", "localhost")
    client_port = int(client.get("port", "50001"))
    client_path = client.get("path", "/wipe_subtitle")
    reconnect_initial_ms = int(client.get("reconnect_initial_ms", "500"))
    reconnect_max_ms = int(client.get("reconnect_max_ms", "5000"))

    return Settings(
        llm_model_name=llm_model_name,
        llm_quantization=llm_quantization,
        llm_n_ctx=llm_n_ctx,
        llm_n_threads=llm_n_threads,
        llm_n_gpu_layers=llm_n_gpu_layers,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        client_host=client_host,
        client_port=client_port,
        client_path=client_path,
        reconnect_initial_ms=reconnect_initial_ms,
        reconnect_max_ms=reconnect_max_ms,
        llm_comment_prompt=llm_comment_prompt,
        llm_comment_max_chars=llm_comment_max_chars,
        llm_comment_context_lines=llm_comment_context_lines,
        llm_comment_include_speaker=llm_comment_include_speaker,
        llm_notify_on_filtered_empty=llm_notify_on_filtered_empty,
    )


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    path = Path(config_path) if config_path else _default_config_path()
    return _load_from_ini(path)