from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests


class AIBackendError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class OllamaBackend:
    """Free local LLM backend using Ollama (https://ollama.com).

    Requires a running Ollama server (default: http://localhost:11434).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout_s: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Return local Ollama model tags (e.g. llama3.1:8b)."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            models = []
            for m in data.get("models", []) or []:
                name = m.get("name")
                if name:
                    models.append(name)
            return sorted(set(models))
        except requests.RequestException as e:
            raise AIBackendError(
                "Could not reach Ollama. Is it installed and running at "
                f"{self.base_url}?\n\nDetails: {e}"
            )

    def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.6,
        top_p: float = 0.95,
        system_prompt_override: Optional[str] = None,
        stop: Optional[Iterable[str]] = None,
        cancel_flag: Optional["CancelFlag"] = None,
    ) -> Iterable[str]:
        """Yield assistant text deltas as they stream."""

        payload: Dict = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.content.strip()
            ],
            "stream": True,
            "options": {
                "temperature": float(temperature),
                "top_p": float(top_p),
            },
        }

        if system_prompt_override:
            # Prepend a system message (Ollama supports system role).
            payload["messages"] = (
                [{"role": "system", "content": system_prompt_override}]
                + payload["messages"]
            )

        if stop:
            payload["options"]["stop"] = list(stop)

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=(5, self.timeout_s),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if cancel_flag is not None and cancel_flag.cancelled:
                        return
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("done") is True:
                        return
                    msg = (obj.get("message") or {}).get("content")
                    if msg:
                        yield msg
        except requests.RequestException as e:
            raise AIBackendError(
                "Ollama request failed. Make sure the Ollama server is running "
                f"and the model '{self.model}' is available.\n\nDetails: {e}"
            )


class CancelFlag:
    """Thread-safe-enough cancel flag for background generation."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True
