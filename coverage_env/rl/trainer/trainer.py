"""Episode collection and PPO optimization loop."""

from __future__ import annotations

from copy import deepcopy

import torch

from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer


class Trainer:
    """Collect complete episodes, update PPO, and print compact episode logs."""

    def __init__(self, environment, agent, *, config: dict | None = None,
                 max_steps: int | None = None, checkpoint_dir=None) -> None:
        self.environment = environment
        self.agent = agent
        self.config = config or {}
        self.rl_config = self.config.get("rl", self.config)
        env_config = self.config.get("environment", {})
        fields_count = max(1, len(environment.fields))
        self._steps_per_field = int(env_config.get("max_episode_steps_per_field", 12))
        self._fixed_max_steps = None if max_steps is None else int(max_steps)
        self.max_steps = int(max_steps or fields_count * self._steps_per_field)
        self.checkpoint_dir = checkpoint_dir
        self.buffer = RolloutBuffer()
        self.history: list[dict] = []
        self.last_rollout_size = 0

    def train(self, num_episode: int) -> list[dict]:
        """Train for ``num_episode`` episodes and return one log row per episode."""
        if num_episode <= 0:
            raise ValueError("num_episode must be positive.")
        self.agent.actor.eval()  # Stable old/new log probabilities; gradients still work.
        for episode in range(1, num_episode + 1):
            state = self.environment.reset()
            # Episode field counts vary under domain randomization, so the
            # truncation cap follows the instance actually drawn by reset().
            if self._fixed_max_steps is None:
                self.max_steps = (
                    max(1, len(self.environment.fields)) * self._steps_per_field
                )
            episode_reward = 0.0
            steps = 0
            last_info = {}
            while steps < self.max_steps:
                output = self.agent.actor(state, environment=self.environment)
                distribution = output["action_distribution"]
                if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0.0:
                    break
                action_index = int(torch.multinomial(distribution, 1).item())
                action = output["actions"][action_index]
                log_prob = torch.log(distribution[action_index].clamp_min(1e-12))
                value = self.agent.value_from_output(output)

                next_state, reward, env_done, info = self.environment.step(action)
                steps += 1
                truncated = steps >= self.max_steps and not env_done
                done = bool(env_done or truncated)
                next_graph = self.agent.actor.graph_builder.build_graph(self.environment)
                next_value = 0.0 if done else self.agent.value_for_graph(next_graph, self.environment)

                state_context = {
                    "observation": deepcopy(state),
                    "graph": _clone_graph(output["graph"]),
                    "actions": list(output["actions"]),
                    "mask": output["mask"].detach().cpu().tolist(),
                }
                next_context = {
                    "observation": deepcopy(next_state),
                    "graph": _clone_graph(next_graph),
                }
                self.buffer.add(
                    state_context, action, reward, next_context, done,
                    log_prob.detach().item(), value.detach().item(), next_value,
                )
                episode_reward += float(reward)
                last_info = info
                state = next_state
                if done:
                    break

            self.last_rollout_size = len(self.buffer)
            ppo_stats = self.agent.update(self.buffer)
            self.buffer.clear()
            summary = self.environment.get_summary()
            success = len(summary["completed_fields"]) == len(self.environment.fields)
            row = {
                "episode": episode,
                "steps": steps,
                "reward": episode_reward,
                "coverage_ratio": summary["coverage_ratio"],
                "total_distance": summary["total_distance"],
                "energy_consumption": summary["total_energy"],
                "remaining_energy": summary["remaining_energy"],
                "success": success,
                "last_action": last_info.get("selected_action"),
                "last_workload": last_info.get("estimated_workload"),
                **ppo_stats,
                "rollout_size": self.last_rollout_size,
            }
            self.history.append(row)
            print(
                f"Episode {episode}/{num_episode} | steps={steps} "
                f"reward={episode_reward:.3f} coverage={summary['coverage_ratio']:.1%} "
                f"distance={summary['total_distance']:.1f}m "
                f"energy={summary['total_energy']:.1f}Wh "
                f"actor_loss={ppo_stats['actor_loss']:.5f} "
                f"critic_loss={ppo_stats['critic_loss']:.5f}"
            )
            if self.checkpoint_dir is not None:
                from pathlib import Path

                path = Path(self.checkpoint_dir)
                path.mkdir(parents=True, exist_ok=True)
                self.agent.save_checkpoint(path / f"ppo_episode_{episode}.pt")
        return list(self.history)


def _clone_graph(graph):
    """Keep a detached graph snapshot suitable for rollout replay."""
    if hasattr(graph, "clone"):
        snapshot = graph.clone()
        if hasattr(snapshot, "cpu"):
            snapshot = snapshot.cpu()
        return snapshot
    return deepcopy(graph)
