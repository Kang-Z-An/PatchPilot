from __future__ import annotations

import json
import urllib.error
import urllib.request


class OpenAICompatibleModel:
    """Minimal client for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 120,
        temperature: float = 0.0,
        thinking_mode: str | None = None,
    ) -> None:
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode must be enabled, disabled, or unset")
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.thinking_mode = thinking_mode

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.thinking_mode is not None:
            payload["thinking"] = {"type": self.thinking_mode}
        body = json.dumps(payload).encode("utf-8")
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
