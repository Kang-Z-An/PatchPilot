from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)


class Model(Protocol):
    """Small model boundary so providers and test doubles are interchangeable."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Return one JSON action as text."""


JSONValue = dict[str, Any]
