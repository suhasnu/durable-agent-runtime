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