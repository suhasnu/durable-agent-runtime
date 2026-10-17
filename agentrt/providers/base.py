"""Provider interface.

One method. Everything vendor-specific lives behind it, so the graph layer
never sees a vendor's JSON and a trace recorded against one model can be
replayed against another.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..types import Completion, Message


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> Completion: ...