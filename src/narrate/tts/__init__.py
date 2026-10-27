"""TTS engine registry.

Adding an engine means writing one module and adding one entry to ``ENGINES``.
"""

from __future__ import annotations

from narrate.tts.base import (
    EngineUnavailableError,
    SynthesisEngine,
    SynthesisError,
    Voice,
)
from narrate.tts.cache import AudioCache

__all__ = [
    "AudioCache",
    "EngineUnavailableError",
    "SynthesisEngine",
    "SynthesisError",
    "Voice",
    "ENGINES",
    "get_engine",
]

#: Engine name -> factory. Imports are deferred so that listing engines, or
#: using one, never pays the import cost of the others' model stacks.
ENGINES: dict[str, str] = {
    "kokoro": "narrate.tts.kokoro_engine:KokoroEngine",
}


def get_engine(name: str = "kokoro", **kwargs) -> SynthesisEngine:
    """Instantiate an engine by name."""
    if name not in ENGINES:
        raise ValueError(
            f"Unknown engine {name!r}. Available: {', '.join(sorted(ENGINES))}"
        )

    import importlib

    module_path, _, class_name = ENGINES[name].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**kwargs)
