from __future__ import annotations

from typing import Any, List, Deque, Tuple
import asyncio

from llama_cpp import Llama
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download, list_repo_files
from collections import deque

from .base import LLMProvider
from ...schemas import LlmRequest
from ...config import Settings, get_settings


class LocalLlamaProvider(LLMProvider):
    name: str = "local-llama"

    def __init__(self, llm: Llama) -> None:
        self._llm = llm
        # 履歴（過去N-1件）を保持
        s = get_settings()
        self._history_max_total_lines: int = max(0, getattr(s, "llm_comment_context_lines", 0))
        self._history_prev_capacity: int = max(0, self._history_max_total_lines - 1)
        self._include_speaker: bool = bool(getattr(s, "llm_comment_include_speaker", True))
        self._history: Deque[Tuple[str | None, str]] = deque(maxlen=self._history_prev_capacity)
        # 次回破棄機能は現在未使用（常時オフ）

    @classmethod
    def from_env(cls, settings: Settings | None = None) -> "LocalLlamaProvider":
        s = settings or get_settings()
        if not s.llm_model_name:
            raise RuntimeError("llm.model_name is required")

        base = s.llm_model_name
        quant = s.llm_quantization
        safe = base.replace(" ", "-").lower()
        candidates = [
            f"./{safe}.{quant}.gguf",
            f"./models/{safe}.{quant}.gguf",
            f"./Models/{safe}.{quant}.gguf",
            f"{str(Path.home())}/Models/{safe}.{quant}.gguf",
        ]
        logger = logging.getLogger("local_llama")
        logger.info("llm: model_name=%s, quantization=%s", base, quant)
        logger.info("llm: local candidates=%s", candidates)
        model_path = None
        for p in candidates:
            if Path(p).exists():
                model_path = p
                break
        if not model_path:
            base = s.llm_model_name
            quant = s.llm_quantization
            # Known repository candidates for requested models
            repo_map: dict[str, List[str]] = {
                "qwen2.5-3b-instruct": [
                    "TheBloke/Qwen2.5-3B-Instruct-GGUF",
                    "bartowski/Qwen2.5-3B-Instruct-GGUF",
                    "Qwen/Qwen2.5-3B-Instruct-GGUF",
                ],
                "qwen2-7b-instruct": [
                    "TheBloke/Qwen2-7B-Instruct-GGUF",
                    "bartowski/Qwen2-7B-Instruct-GGUF",
                    "Qwen/Qwen2-7B-Instruct-GGUF",
                ],
                "llama-3.2-3b-instruct": [
                    "bartowski/Llama-3.2-3B-Instruct-GGUF",
                ],
                "llama-3-8b-instruct": [
                    "bartowski/Llama-3-8B-Instruct-GGUF",
                ],
                "elyza-japanese-llama-2-7b-instruct": [
                    "elyza/ELYZA-japanese-Llama-2-7b-instruct-GGUF",
                    "TheBloke/ELYZA-japanese-Llama-2-7B-instruct-GGUF",
                ],
                "llama-3.1-8b-instruct": [
                    "bartowski/Llama-3.1-8B-Instruct-GGUF",
                ],
                "gemma-2-2b": [
                    "bartowski/gemma-2-2b-GGUF",
                ],
                "gemma-2-2b-it": [
                    "bartowski/gemma-2-2b-it-GGUF",
                ],
            }
            pretty = base.replace("-", " ")
            title_case = "-".join([w[:1].upper() + w[1:] if w else w for w in pretty.split()]).replace(" ", "-")
            generic_candidates = [
                f"TheBloke/{title_case}-GGUF",
                f"bartowski/{title_case}-GGUF",
                f"TheBloke/{base}-GGUF",
                f"bartowski/{base}-GGUF",
            ]
            repo_candidates = repo_map.get(base, []) + generic_candidates
            logger.info("llm: repo candidates=%s", repo_candidates)

            for repo_id in repo_candidates:
                try:
                    logger.info("llm: listing files in %s", repo_id)
                    files = list_repo_files(repo_id)
                except Exception as ex:  # noqa: BLE001
                    logger.info("llm: list failed for %s: %s", repo_id, ex)
                    continue
                quant_norm = quant.lower()
                ggufs = [
                    f for f in files
                    if f.lower().endswith(".gguf") and quant_norm in f.lower()
                ]
                # Prefer files containing base token and shorter name
                ggufs.sort(key=lambda x: (base not in x.lower(), len(x)))
                if not ggufs:
                    logger.info("llm: no *.%s.gguf in %s (files=%s)", quant, repo_id, files[:10])
                    continue
                filename = ggufs[0]
                logger.info("downloading model from %s: %s", repo_id, filename)
                try:
                    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
                    model_path = downloaded
                    break
                except Exception as ex:  # noqa: BLE001
                    logger.warning("download failed for %s/%s: %s", repo_id, filename, ex)

        if not model_path:
            raise RuntimeError("llm.model_name did not resolve locally and download also failed")

        # n_threads=0 -> auto
        llm = Llama(
            model_path=model_path,
            n_ctx=s.llm_n_ctx,
            n_threads=None if s.llm_n_threads <= 0 else s.llm_n_threads,
            n_gpu_layers=s.llm_n_gpu_layers,
            verbose=False,
        )
        return cls(llm)

    async def generate_comment(self, request: LlmRequest) -> str:
        subtitle: str = request.payload.get("text") or request.payload.get("subtitle") or ""
        speaker: str | None = request.payload.get("speaker")
        settings = get_settings()
        # 出力に接頭辞やラベルが混じらないように明確に指示
        # 文脈: context_override が来たらそれを優先し、無ければ履歴（過去N-1件）+現在1件
        context_override: str | None = request.payload.get("context_override")
        if context_override:
            context_line = context_override
        elif self._history_max_total_lines > 1 and self._history:
            lines: List[str] = []
            for spk, sub in list(self._history):
                if self._include_speaker and spk:
                    lines.append(f"[{spk}] {sub}")
                else:
                    lines.append(sub)
            # 現在分
            if self._include_speaker and speaker:
                lines.append(f"[{speaker}] {subtitle}")
            else:
                lines.append(subtitle)
            context_line = "\n".join(lines)
        else:
            context_line = f"[{speaker}] {subtitle}" if (self._include_speaker and speaker) else subtitle
        # 設定でプロンプトを差し替え可能（{context} を置換）
        if settings.llm_comment_prompt:
            prompt = settings.llm_comment_prompt.replace("{context}", context_line)
        else:
            prompt = (
                "次の発話に対して、日本語で短い相槌・一言コメントを1行だけ返してください。\n"
                "今の一言:から始まる複数行の返答は禁止\n"
                "出力はコメント本文1行のみ。接頭辞・ラベル・引用・記号は一切付けない。最大50文字。\n"
                f"発話: {context_line}\n"
                "コメント:"
            )

        # llama.cpp は同期APIのため、スレッドで実行
        # create_completion は OpenAI 風の choices[0].text を返す
        # 次回破棄機能は無効化中
        out: Any = await asyncio.to_thread(
            self._llm.create_completion,
            prompt=prompt,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            # 先頭にラベルが出始めたら即停止させる
            stop=["Human:", "User:", "Assistant:", "assistant:", "アシスタント:", "出力:", "出力：", "Output:", "コメント:", "コメント："],
        )
        text = (out.get("choices", [{}])[0].get("text") or "").strip()
        raw_text = text
        # 次回破棄は行わない
        filter_hits: List[str] = []
        # 念のため、先頭行のみ・余計な接頭辞を除去
        if "\n" in text:
            # 複数行生成に対しては固定の注意を内部通知
            try:
                logging.getLogger("local_llama").info("[notify] advise=複数行の返答は禁止です")
                _ = await asyncio.to_thread(
                    self._llm.create_completion,
                    prompt="複数行の返答は禁止です",
                    max_tokens=8,
                    temperature=0.0,
                    stop=["\n"],
                )
            except Exception:
                pass
            text = text.splitlines()[0].strip()
        # 先頭に付与されがちなラベルだけを除去（本文中は残す）
        for prefix in ("コメント:", "コメント：", "comment:", "応答:", "応答：", "assistant:", "Assistant:", "アシスタント:", "Human:", "User:", "出力:", "出力：", "Output:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].lstrip(" ：:　")
                filter_hits.append(f"prefix:{prefix}")
                break
        # 文中にラベルチェーンが混じった場合はその手前で切る（安全側）
        cut_markers = (" Human:", " User:", " Assistant:", " アシスタント:", " 出力:", " 出力：", " コメント:", " コメント：", " Output:")
        for marker in cut_markers:
            if marker in text:
                text = text.split(marker, 1)[0].rstrip()
                filter_hits.append(f"cut:{marker.strip()}")
                break
        # --- 追加のサニタイズ ---
        # 1) URL/ドメイン/コード記法/ハッシュタグ/回答ラベル系の除去・打ち切り
        #    （日本語の口語相槌に不要なものを機械的に落とす）
        patterns_to_cut = (
            "http://", "https://", ".com", ".net", ".org", "#", "```", "/*", "*/", "//", "noqa", "解答:", "回答:", "Answer:", "Question:", "请", "問題:", "問題："
        )
        for p in patterns_to_cut:
            if p in text:
                text = text.split(p, 1)[0].rstrip()
                filter_hits.append(f"pattern:{p}")
                break

        # 3) 句点で切る（最初の「。」まで）
        if "。" in text:
            # 句点がある場合は句点までで確定（丸めは原則しない）
            text = text.split("。", 1)[0] + "。"
            # ただし、極端に長い場合だけ最大文字数で丸める
            max_chars = settings.llm_comment_max_chars if hasattr(settings, "llm_comment_max_chars") else 50
            if len(text) > max_chars:
                text = text[:max_chars]
        else:
            # 4) 句点が無い場合のみ、最大文字数で丸める
            max_chars = settings.llm_comment_max_chars if hasattr(settings, "llm_comment_max_chars") else 50
            if len(text) > max_chars:
                text = text[:max_chars]

        # --- サニタイズここまで ---
        # 追加: 記号のみ/日本語非含有を空扱い（Python reは\p{...}未対応のため手動判定）
        import re as _re
        import string as _string
        _jp_punct = set("。、「」『』（）()…・〜ー！？：；〔〕［］｛｝〈〉《》—―・")
        _punct_set = set(_string.punctuation) | _jp_punct
        if text and all((ch.isspace() or ch in _punct_set) for ch in text):
            filter_hits.append("only_punct")
            text = ""
        if text and not _re.search(r"[ぁ-んァ-ヶ一-龠]", text):
            filter_hits.append("no_japanese")
            text = ""

        if not text:
            logger = logging.getLogger("local_llama")
            logger.info("[filter] sanitized empty; raw='%s'; hits=[%s]", raw_text, ", ".join(filter_hits))
            # 注意文（AIにも通知し、返答は破棄）
            notify_prompt = (
                "さっきのコメントはフィルタで削除されました。"
                "文章の先頭で改行を入れるな。"
                "句読点や記号のみは出さない。必ず日本語を含める。"
                "URLやラベルや記号やコードっぽいのは入れるな。"
                "日本語の口語で短い相槌または一言だけを返してください。"
                "出力は本文のみ。新しい会話・説明・要約・登場人物名・括弧や記号やラベルや改行を付けない。"
            )
            logger.info("[notify] advise=%s", notify_prompt)
            try:
                _ = await asyncio.to_thread(
                    self._llm.create_completion,
                    prompt=notify_prompt,
                    max_tokens=8,
                    temperature=0.0,
                    stop=["\n"],
                )
            except Exception:
                pass
            # 次回破棄は行わない

        # ログ: raw -> sanitized 差分
        if text != raw_text:
            logging.getLogger("local_llama").info("[filter] raw='%s' -> sanitized='%s'", raw_text, text)

        # 履歴を更新（現在の字幕を末尾に追加）
        if self._history_prev_capacity > 0:
            self._history.append((speaker, subtitle))
        # 次回破棄は行わない
        return text or ""

