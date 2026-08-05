from __future__ import annotations

import json
import urllib.error
import urllib.request


class OpenAICompatibleModel:
    """Minimal client for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, model: str, api_key: str, base_url: str, timeout_seconds: int = 120) -> None:
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach model API: {exc.reason}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {payload}") from exc
