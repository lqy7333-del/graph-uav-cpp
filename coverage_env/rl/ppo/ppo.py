"""Actor-critic PPO agent for masked heterogeneous-graph actions."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.ppo.loss import compute_advantage


class ValueNetwork(nn.Module):
    """Critic that pools Field/Boundary embeddings and global context."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, field_embedding, boundary_embedding, global_embedding):
        """Return scalar values with shape ``[batch, 1]``."""
        field_pool = self._pool(field_embedding)
        boundary_pool = self._pool(boundary_embedding)
        global_pool = global_embedding.reshape(-1, self.hidden_dim)
        return self.network(torch.cat((field_pool, boundary_pool, global_pool), dim=-1))

    def _pool(self, embeddings):
        if embeddings.shape[0] == 0:
            return embeddings.new_zeros((1, self.hidden_dim))
        return embeddings.mean(dim=0, keepdim=True)


class PPOAgent:
    """Optimize a ``CoveragePolicy`` actor and an independent graph-value critic."""

    def __init__(self, actor, config: dict | None = None, *, device=None) -> None:
        config = config or {}
        self.config = config.get("ppo", config)
        self.actor = actor
        self.device = torch.device(device or next(actor.parameters()).device)
        self.actor.to(self.device)
        self.critic = ValueNetwork(actor.encoder.hidden_dim).to(self.device)
        learning_rate = float(self.config.get("learning_rate", 3e-4))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = float(self.config.get("gamma", 0.99))
        self.gae_lambda = float(self.config.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.config.get("clip_ratio", 0.2))
        self.entropy_coef = float(self.config.get("entropy_coef", 0.01))
        self.value_coef = float(self.config.get("value_coef", 0.5))
        self.epochs = int(self.config.get("epochs", 4))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        self.update_count = 0

    def value_from_output(self, output) -> torch.Tensor:
        """Evaluate the critic from an actor forward result."""
        embeddings = output["encoder_output"]
        return self.critic(
            embeddings["field_embedding"].detach(),
            embeddings["boundary_embedding"].detach(),
            output["global_embedding"].detach(),
        ).squeeze(-1)

    @torch.no_grad()
    def value_for_graph(self, graph) -> float:
        """Estimate V(s) for a saved graph without generating actions or masks."""
        was_training = self.actor.training
        self.actor.eval()
        embeddings, global_embedding = self.actor.encode_graph(graph)
        value = self.critic(
            embeddings["field_embedding"], embeddings["boundary_embedding"],
            global_embedding,
        ).reshape(()).item()
        self.actor.train(was_training)
        return float(value)

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run clipped PPO policy and value updates on the collected rollout."""
        transitions = list(buffer.transitions)
        if not transitions:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "updates": 0}
        advantages, returns = compute_advantage(
            [row.reward for row in transitions],
            [row.value for row in transitions],
            [row.next_value for row in transitions],
            [row.done for row in transitions],
            self.gamma, self.gae_lambda, device=self.device,
        )
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses, critic_losses, entropies = [], [], []
        for _epoch in range(max(1, self.epochs)):
            permutation = torch.randperm(len(transitions)).tolist()
            batch_size = max(1, self.batch_size)
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start:start + batch_size]
                log_probs, entropy_rows, values = [], [], []
                for index in indices:
                    transition = transitions[index]
                    policy_state = transition.state
                    output = self.actor(
                        policy_state["observation"],
                        graph=policy_state["graph"],
                        candidate_actions=policy_state["actions"],
                        feasible_mask=policy_state["mask"],
                    )
                    action_index = policy_state["actions"].index(transition.action)
                    probabilities = output["action_distribution"]
                    log_probs.append(torch.log(probabilities[action_index].clamp_min(1e-12)))
                    valid_probabilities = probabilities[output["mask"]]
                    entropy_rows.append(-(
                        valid_probabilities * torch.log(valid_probabilities.clamp_min(1e-12))
                    ).sum())
                    values.append(self.value_from_output(output).reshape(()))

                batch_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                current_log_probs = torch.stack(log_probs)
                entropy = torch.stack(entropy_rows).mean()
                current_values = torch.stack(values)
                old_log_probs = torch.as_tensor(
                    [transitions[index].log_prob for index in indices],
                    dtype=torch.float32, device=self.device,
                )
                ratios = torch.exp(current_log_probs - old_log_probs)
                batch_advantages = advantages[batch_indices]
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(
                    ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.entropy_coef * entropy
                critic_loss = F.mse_loss(current_values, returns[batch_indices])

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().item()))
                critic_losses.append(float(critic_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))

        self.update_count += 1
        return {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "entropy": sum(entropies) / len(entropies),
            "updates": len(actor_losses),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save actor, critic, both optimizers, and update count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path, *, map_location=None) -> None:
        """Restore a checkpoint created by ``save_checkpoint``."""
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
