"""Deterministic evaluation for trained coverage policies."""

from __future__ import annotations

import torch

from coverage_env.rl.decoder.action_decoder import CoveragePolicy


class Evaluator:
    """Run greedy episodes and summarize reward, coverage, distance, and energy."""

    def __init__(self, environment, actor: CoveragePolicy, *, max_steps=None) -> None:
        self.environment = environment
        self.actor = actor
        self.max_steps = max_steps or max(1, len(environment.fields) * 12)

    @torch.no_grad()
    def evaluate(self, num_episode: int = 1) -> dict:
        """Evaluate deterministic greedy actions over independent resets."""
        if num_episode <= 0:
            raise ValueError("num_episode must be positive.")
        was_training = self.actor.training
        self.actor.eval()
        episode_rows = []
        for _ in range(num_episode):
            state = self.environment.reset()
            reward_sum = 0.0
            for _step in range(self.max_steps):
                output = self.actor(state, environment=self.environment)
                distribution = output["action_distribution"]
                if distribution.numel() == 0 or float(distribution.sum()) <= 0.0:
                    break
                action = self.actor.greedy_action(output)
                next_state, reward, done, _info = self.environment.step(action)
                reward_sum += float(reward)
                state = next_state
                if done:
                    break
            summary = self.environment.get_summary()
            episode_rows.append({
                "reward": reward_sum,
                "coverage_ratio": summary["coverage_ratio"],
                "total_distance": summary["total_distance"],
                "energy_consumption": summary["total_energy"],
                "success": len(summary["completed_fields"]) == len(self.environment.fields),
            })
        self.actor.train(was_training)
        count = len(episode_rows)
        metrics = {
            key: sum(row[key] for row in episode_rows) / count
            for key in ("reward", "coverage_ratio", "total_distance", "energy_consumption")
        }
        metrics["success_rate"] = sum(row["success"] for row in episode_rows) / count
        metrics["episodes"] = episode_rows
        return metrics
