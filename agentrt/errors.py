"""Error taxonomy.

The runtime needs one answer about every failure: retry, or stop. A flat
exception hierarchy cannot give it, so the classification lives in the type.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base for everything this runtime raises."""


class RetryableError(AgentError):
    """Transient. The same call, made again, might succeed."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalError(AgentError):
    """Permanent. Retrying the same call cannot help."""

class UncertainSideEffect(AgentError):
    """A side-effecting tool started but its outcome was never recorded.

    The write-ahead log holds a `started` record with no matching `completed`
    record. The runtime cannot know whether the effect happened, so it refuses
    to guess. See docs/durability.md.
    """

    def __init__(self, tool_name: str, call_id: str):
        super().__init__(
            f"tool {tool_name!r} (call {call_id}) may have executed before the crash; "
            f"outcome unknown. Mark the tool idempotent, or resolve this run manually."
        )
        self.tool_name = tool_name
        self.call_id = call_id
