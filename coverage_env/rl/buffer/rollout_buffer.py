"""On-policy transition storage for PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass
class RolloutTransition:
    """One state/action transition and the policy context needed for replay."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    log_prob: float
    value: float
    next_value: float = 0.0


class RolloutBuffer:
    """Store complete on-policy episodes until the PPO update consumes them."""

    def __init__(self) -> None:
        self.transitions: list[RolloutTransition] = []

    def add(
        self, state, action, reward, next_state, done, log_prob, value,
        next_value=0.0,
    ) -> None:
        """Append one transition, converting scalar values to plain Python types."""
        self.transitions.append(RolloutTransition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            log_prob=float(log_prob),
            value=float(value),
            next_value=float(next_value),
        ))

    def clear(self) -> None:
        """Remove every stored transition."""
        self.transitions.clear()

    def sample(self, batch_size: int | None = None, *, rng=None) -> list[RolloutTransition]:
        """Return a random minibatch, or all transitions in random order."""
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        source = rng or random
        if batch_size is None or batch_size >= len(self.transitions):
            return source.sample(self.transitions, len(self.transitions))
        return source.sample(self.transitions, batch_size)

    def __len__(self) -> int:
        return len(self.transitions)
