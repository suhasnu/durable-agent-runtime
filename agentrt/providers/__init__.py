from .base import Provider
from .scripted import ScriptedProvider, calls, calls_many, rate_limited, says

__all__ = [
    "Provider",
    "ScriptedProvider",
    "calls",
    "calls_many",
    "rate_limited",
    "says",
]


def __getattr__(name: str):
    # Imported lazily so the package works without an API key present.
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(name)
