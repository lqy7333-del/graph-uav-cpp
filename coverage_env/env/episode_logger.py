"""In-memory and JSONL logging for environment state transitions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


class EpisodeLogger:
    """Record immutable snapshots of ``state, action, feedback, next_state``."""

    def __init__(self) -> None:
        """Create an empty episode log."""
        self._transitions: list[dict] = []

    @property
    def transitions(self) -> tuple[dict, ...]:
        """Return logged transitions in execution order."""
        return tuple(self._transitions)

    def reset(self) -> None:
        """Clear records at the start of a new episode."""
        self._transitions.clear()

    def log_transition(self, before: dict, action: Any, result: dict, after: dict) -> dict:
        """Store a JSON-compatible snapshot and return that record."""
        transition = {
            "step": len(self._transitions),
            "before_state": self._jsonable(before),
            "action": self._jsonable(action),
            "result": self._jsonable(result),
            "after_state": self._jsonable(after),
        }
        self._transitions.append(transition)
        return transition

    def save_jsonl(self, path: str | Path, *, append: bool = False) -> Path:
        """Write one JSON object per line and return the output path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with output.open(mode, encoding="utf-8") as stream:
            for transition in self._transitions:
                stream.write(json.dumps(transition, ensure_ascii=False) + "\n")
        return output

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """Convert dataclasses and coordinate containers to JSON values."""
        if is_dataclass(value) and not isinstance(value, type):
            return cls._jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): cls._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonable(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)
