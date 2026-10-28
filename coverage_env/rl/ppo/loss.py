"""PPO returns and generalized advantage estimation."""

from __future__ import annotations

import torch


def compute_advantage(
    rewards, values, next_values, dones, gamma: float = 0.99,
    gae_lambda: float = 0.95, *, device=None, truncated=None,
):
    """Compute GAE advantages and value targets for one ordered rollout.

    ``delta_t = reward_t + gamma * next_value_t * (1-done_t) - value_t``.
    Returns ``(advantages, returns)`` as one-dimensional tensors.
    """
    reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(-1)
    value_t = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1)
    next_value_t = torch.as_tensor(next_values, dtype=torch.float32, device=device).reshape(-1)
    done_t = torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(-1)
    boundary_t = done_t.bool()
    if truncated is not None:
        trunc_t = torch.as_tensor(truncated, dtype=torch.bool, device=device).reshape(-1)
        if len(trunc_t) != len(done_t):
            raise ValueError("truncated and dones must have equal lengths.")
        boundary_t = boundary_t | trunc_t
    if not (len(reward_t) == len(value_t) == len(next_value_t) == len(done_t)):
        raise ValueError("rewards, values, next_values, and dones must have equal lengths.")
    advantages = torch.zeros_like(reward_t)
    gae = torch.zeros((), dtype=reward_t.dtype, device=reward_t.device)
    for index in range(len(reward_t) - 1, -1, -1):
        nonterminal = 1.0 - done_t[index]
        delta = reward_t[index] + gamma * next_value_t[index] * nonterminal - value_t[index]
        # Time limits bootstrap V(next), but must not leak GAE into the next episode.
        gae = delta + gamma * gae_lambda * (~boundary_t[index]).float() * gae
        advantages[index] = gae
    return advantages, advantages + value_t
