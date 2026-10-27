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


class TruncatedResponse(FatalError):
    """The model stopped because it hit the token ceiling, mid-sentence.

    Fatal rather than retryable: the same request would be cut off at the same
    place. Dangerous rather than merely incomplete, because a tool call cut off
    halfway can still parse, so the arguments that survived would be executed
    as if the model had asked for them.
    """

    def __init__(self, max_tokens: int):
        super().__init__(
            f"response truncated at max_tokens={max_tokens}; resume this run "
            f"with a higher limit rather than acting on a partial response"
        )
        self.max_tokens = max_tokens


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
