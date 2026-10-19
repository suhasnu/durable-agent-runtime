from .anthropic import AnthropicProvider
from .base import Provider
from .scripted import ScriptedProvider, calls, says

__all__ = ["AnthropicProvider", "Provider", "ScriptedProvider", "calls", "says"]

def __getattr__(name: str):
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    raise AttributeError(name)