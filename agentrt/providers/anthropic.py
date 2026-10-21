"""Anthropic Messages API adapter.

The translation in and out of the vendor shape is the only code in this
package that knows what a tool_use block is.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ..errors import FatalError, RetryableError
from ..types import Completion, Message, ToolCall, Usage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_attempts: int = 4,
    ):
        if max_attempts < 1:
          raise ValueError("max_attempts must be at least 1")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise FatalError("ANTHROPIC_API_KEY is not set")

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 2048,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _to_wire(messages),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools

        return _from_wire(self._post_with_retry(body))

    def _post_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return self._post(body)
            except RetryableError as exc:
                last = exc
                if attempt == self.max_attempts - 1:
                    break
                time.sleep(exc.retry_after or delay)
                delay *= 2
        raise last

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": API_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code in RETRYABLE_STATUS:
                raise RetryableError(
                    f"HTTP {exc.code}: {detail}",
                    retry_after=_parse_retry_after(exc.headers.get("retry-after")),
                ) from exc
            raise FatalError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RetryableError(f"connection failed: {exc.reason}") from exc
        except (TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
          raise RetryableError(f"{type(exc).__name__}: {exc}") from exc

def _parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait, capped.

    The header may legally be an HTTP date rather than a number, and a server
    is free to ask for an hour. Neither should stall the process.
    """
    if not value:
        return None
    try:
        return min(float(value), 30.0)
    except ValueError:
        return None


def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Our flat Message list becomes the API's content-block structure.

    Tool results are the awkward part: the API carries them on a `user` turn,
    not a `tool` turn, and consecutive results must merge into one turn.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "assistant" and not msg.content and not msg.tool_calls:
            continue

        if msg.role == "tool":
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    **({"is_error": True} if r.is_error else {}),
                }
                for r in msg.tool_results
            ]
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": "user", "content": blocks})
            continue

        if msg.role == "assistant" and msg.tool_calls:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in msg.tool_calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue

        out.append({"role": msg.role, "content": msg.content})
    return out


def _from_wire(raw: dict[str, Any]) -> Completion:
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {}))
            )

    usage = raw.get("usage", {})
    return Completion(
        message=Message(role="assistant", content="".join(text_parts), tool_calls=calls),
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ),
        stop_reason=raw.get("stop_reason", "end_turn"),
        model=raw.get("model", ""),
    )
