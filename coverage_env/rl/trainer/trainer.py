"""Episode collection and PPO optimization loop."""

from __future__ import annotations

from copy import deepcopy

import torch

from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.trainer.episode import nearest_neighbor_return, current_potential, learning_reward
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state


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
        self._next_episode = 1
        self._training_horizon = None

    def train(self, num_episode: int) -> list[dict]:
        """Train for ``num_episode`` episodes and return one log row per episode."""
        if num_episode <= 0:
            raise ValueError("num_episode must be positive.")
        self.agent.actor.eval()  # Stable old/new log probabilities; gradients still work.
        if self._training_horizon is not None and self._training_horizon != num_episode:
            raise ValueError("Exact resume requires the original episode horizon.")
        self._training_horizon = num_episode
        for episode in range(self._next_episode, num_episode + 1):
            state = self.environment.reset()
            if self.agent.config.get("advantage_mode", "gae") == "broadcast_nn":
                baseline, _ = nearest_neighbor_return(self.environment, self.config, self.agent.gamma)
                self.agent.nn_baselines.append(baseline)
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

                before = current_potential(self.environment, self.config)
                next_state, reward, env_done, info = self.environment.step(action)
                shaped = learning_reward(self.environment, reward, before, env_done,
                                         self.config, self.agent.gamma)
                steps += 1
                truncated = steps >= self.max_steps and not env_done
                done = bool(env_done or truncated)
                next_graph = self.agent.actor.graph_builder.build_graph(self.environment)
                next_value = 0.0 if env_done else self.agent.value_for_graph(next_graph, self.environment)

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
                    state_context, action, shaped, next_context, env_done,
                    log_prob.detach().item(), value.detach().item(), next_value,
                    truncated=truncated,
                )
                episode_reward += float(reward)
                last_info = info
                state = next_state
                if done:
                    break

            self.last_rollout_size = len(self.buffer)
            self.agent.set_training_progress((episode - 1) / max(1, num_episode - 1))
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
            self._next_episode = episode + 1
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
                self.save_checkpoint(path / f"ppo_episode_{episode}.pt")
        return list(self.history)

    def save_checkpoint(self, path):
        """Save the whole episode-boundary runtime, not only model weights."""
        from pathlib import Path
        payload = self.agent.checkpoint_payload()
        payload["trainer_runtime"] = {
            "environment": deepcopy(self.environment), "rng": capture_rng_state(),
            "next_episode": self._next_episode, "horizon": self._training_horizon,
            "fixed_max_steps": self._fixed_max_steps, "config": deepcopy(self.config),
            "history": deepcopy(self.history),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path):
        """Resume an episode-boundary snapshot with the original configuration."""
        saved = torch.load(path, map_location="cpu", weights_only=False)
        runtime = saved.get("trainer_runtime")
        if runtime is None:
            raise ValueError("No Trainer runtime in checkpoint; load weights through PPOAgent for fine-tuning.")
        if runtime["config"] != self.config or runtime["fixed_max_steps"] != self._fixed_max_steps:
            raise ValueError("Exact Trainer resume requires unchanged config and max_steps.")
        self.agent.load_checkpoint(path, restore_rng=False)
        self.environment = runtime["environment"]
        self.agent.actor.environment = self.environment
        self._next_episode = runtime["next_episode"]
        self._training_horizon = runtime["horizon"]
        self.history = runtime["history"]
        restore_rng_state(runtime["rng"])


def _clone_graph(graph):
    """Keep a detached graph snapshot suitable for rollout replay."""
    if hasattr(graph, "clone"):
        snapshot = graph.clone()
        if hasattr(snapshot, "cpu"):
            snapshot = snapshot.cpu()
        return snapshot
    return deepcopy(graph)
