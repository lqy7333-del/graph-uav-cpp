"""PPO optimization components."""

from coverage_env.rl.ppo.ppo import PPOAgent, ValueNetwork
from coverage_env.rl.ppo.loss import compute_advantage

__all__ = ["PPOAgent", "ValueNetwork", "compute_advantage"]
