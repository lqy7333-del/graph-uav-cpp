"""Actor-critic PPO agent for masked heterogeneous-graph actions."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.ppo.loss import compute_advantage


class ValueNetwork(nn.Module):
    """Critic mapping the critic features (learned ``z_G`` + raw state) to a scalar."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, critic_features):
        """Return scalar values with shape ``[batch, 1]``."""
        return self.network(critic_features.reshape(-1, self.input_dim))


class PPOAgent:
    """Optimize a ``CoveragePolicy`` actor and an independent graph-value critic."""

    def __init__(self, actor, config: dict | None = None, *, device=None) -> None:
        config = config or {}
        self.config = config.get("ppo", config)
        self.actor = actor
        self.device = torch.device(device or next(actor.parameters()).device)
        self.actor.to(self.device)
        # The critic consumes ``z_G`` plus the raw UAV/depot/site features. The
        # raw part matters: ``GlobalStateEncoder`` is trained only by the actor
        # and detached before the critic, so the layout information the value
        # function depends on must not be reachable *only* through it.
        critic_input_dim = actor.encoder.hidden_dim
        global_encoder = getattr(actor, "global_state_encoder", None)
        if global_encoder is not None:
            critic_input_dim += (
                global_encoder.uav_dim
                + global_encoder.depot_dim
                + global_encoder.site_dim
            )
        self.critic = ValueNetwork(critic_input_dim, actor.encoder.hidden_dim).to(self.device)
        learning_rate = float(self.config.get("learning_rate", 3e-4))
        # Each optimizer gets its own rate. Adam moves a parameter by roughly
        # `lr` per step, so `lr x steps` bounds how far a network can travel in a
        # whole run: at the shared 3e-4 over 200 updates (8 minibatches each) the
        # actor can only move ~0.5 in total, and the critic needed 10x to span
        # the return range at all (explained_variance was stuck at ~0 until then).
        actor_learning_rate = float(
            self.config.get("actor_learning_rate", learning_rate)
        )
        critic_learning_rate = float(
            self.config.get("critic_learning_rate", learning_rate)
        )
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.gamma = float(self.config.get("gamma", 0.99))
        self.gae_lambda = float(self.config.get("gae_lambda", 0.95))
        self.clip_ratio = float(self.config.get("clip_ratio", 0.2))
        self.value_coef = float(self.config.get("value_coef", 0.5))
        self.epochs = int(self.config.get("epochs", 4))
        self.batch_size = int(self.config.get("batch_size", 64))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        # Entropy annealing. A constant bonus kept the policy at ~77% of maximum
        # entropy after 400 updates (3.41 nats against log(84)=4.43), which costs
        # reward on the sampled actions it is evaluated with and, worse, floods the
        # advantage with exploration noise -- the ordering signal is only ~2 points
        # against an episode standard deviation of ~2.6.
        self.entropy_coef = float(self.config.get("entropy_coef", 0.01))
        self.entropy_coef_start = self.entropy_coef
        end = self.config.get("entropy_coef_end", None)
        self.entropy_coef_end = None if end is None else float(end)
        self.entropy_anneal_start = float(
            self.config.get("entropy_anneal_start", 0.25)
        )
        # Learning-rate decay. A constant rate keeps injecting gradient noise, so a
        # long run never comes to rest; decaying it lets the policy settle. The
        # factor is driven by the same run-progress fraction as the entropy bonus,
        # so it stretches automatically when `--num-updates` grows.
        self.lr_schedule = str(self.config.get("lr_schedule", "none")).lower()
        self.lr_final_scale = float(self.config.get("lr_final_scale", 0.1))
        self.actor_lr_start = float(actor_learning_rate)
        self.critic_lr_start = float(critic_learning_rate)
        # NOTE: `max_grad_norm` only *rescales* the gradient, and Adam is
        # invariant to a uniform gradient rescale (verified), so clipping does
        # not limit the step size. Do not read `grad_norm == max_grad_norm` as
        # evidence that a network is step-limited; `lr` is the only lever.
        self.update_count = 0
        self._value_bias_initialized = False

    @staticmethod
    def _critic_features(output) -> torch.Tensor:
        """Return the critic's input from an actor forward result.

        Prefers the policy's ``critic_input`` (``z_G`` + raw UAV/depot/site
        features) and falls back to ``global_embedding`` for callers that only
        produce the latter.
        """
        if "critic_input" in output:
            return output["critic_input"]
        return output["global_embedding"]

    def value_from_output(self, output) -> torch.Tensor:
        """Evaluate the critic from an actor forward result."""
        return self.critic(self._critic_features(output).detach()).squeeze(-1)

    @torch.no_grad()
    def value_for_graph(self, graph, environment=None) -> float:
        """Estimate V(s) for a saved graph without generating actions or masks."""
        was_training = self.actor.training
        self.actor.eval()
        _embeddings, _global_embedding, critic_input = self.actor.encode_graph(
            graph, environment=environment
        )
        value = self.critic(critic_input).reshape(()).item()
        self.actor.train(was_training)
        return float(value)

    def set_entropy_progress(self, fraction: float) -> float:
        """Anneal the entropy bonus over the run; return the current coefficient.

        ``fraction`` is the run progress in ``[0, 1]``. The coefficient is held at
        its start value for ``entropy_anneal_start`` of the run and then decays
        linearly to ``entropy_coef_end``, keeping early exploration while letting
        the policy sharpen instead of staying near-uniform.
        """
        if self.entropy_coef_end is None:
            return self.entropy_coef
        fraction = min(1.0, max(0.0, float(fraction)))
        hold = min(1.0, max(0.0, self.entropy_anneal_start))
        scale = 1.0 if fraction <= hold else 1.0 - (fraction - hold) / max(1e-9, 1.0 - hold)
        self.entropy_coef = self.entropy_coef_end + (
            self.entropy_coef_start - self.entropy_coef_end
        ) * scale
        return self.entropy_coef

    def _set_learning_rate_progress(self, fraction: float):
        """Decay both optimisers' learning rates; return the actor's current rate."""
        if self.lr_schedule not in {"linear", "cosine"}:
            return self.actor_lr_start
        import math

        fraction = min(1.0, max(0.0, float(fraction)))
        if self.lr_schedule == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
        else:
            factor = 1.0 - fraction
        scale = self.lr_final_scale + (1.0 - self.lr_final_scale) * factor
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_lr_start * scale
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_lr_start * scale
        return self.actor_lr_start * scale

    def set_training_progress(self, fraction: float) -> dict:
        """Advance every schedule tied to run progress.

        The entropy bonus and the learning rate are the two "temperature-like"
        knobs in this setup: the first sharpens the policy's own distribution, the
        second shrinks the step size. Both are driven by one progress fraction so
        they stay aligned and automatically stretch for longer runs.
        """
        return {
            "entropy_coef": self.set_entropy_progress(fraction),
            "actor_learning_rate": self._set_learning_rate_progress(fraction),
        }

    def update(self, buffer: RolloutBuffer, *, batched: bool = False) -> dict:
        """Run clipped PPO policy and value updates on the collected rollout.

        ``batched=True`` replaces the per-transition encoder forward with a
        single disjoint-union pass per minibatch (see
        ``coverage_env.rl.encoder.batched_encoder``). Log-probs, entropies, and
        values are numerically identical to the default path, so the flag only
        changes speed. It defaults to ``False`` so training behaviour is
        unchanged unless a caller explicitly opts in.
        """
        transitions = list(buffer.transitions)
        if not transitions:
            return self._empty_update_stats()
        advantages, returns = compute_advantage(
            [row.reward for row in transitions],
            [row.value for row in transitions],
            [row.next_value for row in transitions],
            [row.done for row in transitions],
            self.gamma, self.gae_lambda, device=self.device,
        )
        # Diagnostic only: the raw advantage scale, captured *before* the
        # per-update normalization that makes its magnitude uninformative.
        advantage_std = self._tensor_std(advantages)
        self._warm_start_value_bias(returns)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses, critic_losses, entropies = [], [], []
        policy_losses, ratio_stds, clip_fractions = [], [], []
        approx_kls, grad_norms = [], []
        predictions, targets = [], []
        for _epoch in range(max(1, self.epochs)):
            permutation = torch.randperm(len(transitions)).tolist()
            batch_size = max(1, self.batch_size)
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start:start + batch_size]
                if batched:
                    current_log_probs, entropy, current_values = (
                        self._batched_minibatch_rows(transitions, indices)
                    )
                else:
                    current_log_probs, entropy, current_values = (
                        self._per_transition_minibatch_rows(transitions, indices)
                    )

                batch_indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
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
                # Keep the advantage-weighted surrogate separate from the
                # entropy bonus so the log can show which one drives the actor.
                surrogate = torch.minimum(unclipped, clipped).mean()
                actor_loss = -surrogate - self.entropy_coef * entropy
                critic_loss = F.mse_loss(current_values, returns[batch_indices])

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                # ``clip_grad_norm_`` returns the total norm *before* clipping,
                # which is exactly the actor-gradient magnitude we want.
                total_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().item()))
                critic_losses.append(float(critic_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                policy_losses.append(float(-surrogate.detach().item()))
                ratio_stds.append(self._tensor_std(ratios))
                clip_fractions.append(
                    self._tensor_mean(
                        (torch.abs(ratios.detach() - 1.0) > self.clip_ratio).float()
                    )
                )
                approx_kls.append(
                    self._tensor_mean(old_log_probs - current_log_probs)
                )
                grad_norms.append(
                    float(total_norm.detach().item())
                    if torch.is_tensor(total_norm) else float(total_norm)
                )
                predictions.append(current_values.detach())
                targets.append(returns[batch_indices].detach())

        self.update_count += 1
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        return {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "entropy": sum(entropies) / len(entropies),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "advantage_std": advantage_std,
            "return_mean": self._tensor_mean(returns),
            "return_std": self._tensor_std(returns),
            "value_mean": self._tensor_mean(prediction),
            "value_std": self._tensor_std(prediction),
            "explained_variance": self._explained_variance(prediction, target),
            "ratio_std": sum(ratio_stds) / len(ratio_stds),
            "clip_fraction": sum(clip_fractions) / len(clip_fractions),
            "approx_kl": sum(approx_kls) / len(approx_kls),
            "actor_grad_norm": sum(grad_norms) / len(grad_norms),
            "updates": len(actor_losses),
        }

    @staticmethod
    def _empty_update_stats() -> dict:
        """Return zeroed statistics so the log schema is stable."""
        return {
            "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
            "policy_loss": 0.0, "advantage_std": 0.0, "return_mean": 0.0,
            "return_std": 0.0, "value_mean": 0.0, "value_std": 0.0,
            "explained_variance": 0.0, "ratio_std": 0.0,
            "clip_fraction": 0.0, "approx_kl": 0.0, "actor_grad_norm": 0.0,
            "updates": 0,
        }

    def _critic_module(self):
        """Return the critic module, unwrapping DDP if present."""
        return self.critic.module if hasattr(self.critic, "module") else self.critic

    def _warm_start_value_bias(self, returns: torch.Tensor) -> None:
        """Initialise the critic's output bias to the observed mean return.

        The reward is a path cost, so a return is a large negative offset plus a
        much smaller state-dependent part. Adam moves a bias by roughly the
        learning rate per step, so starting from zero the critic spends thousands
        of steps just learning the offset (measured: ~0.5 per update while the
        offset was ~18). That offset error dominates the loss, so the
        state-dependent signal that ``explained_variance`` measures never gets
        fitted. One warm start suffices because the offset is roughly constant.
        """
        if self._value_bias_initialized or returns.numel() == 0:
            return
        final_layer = self._critic_module().network[-1]
        with torch.no_grad():
            final_layer.bias.fill_(float(returns.mean().item()))
        self._value_bias_initialized = True

    @staticmethod
    def _tensor_mean(values: torch.Tensor) -> float:
        """Return the mean of ``values`` as a float (0.0 when empty)."""
        if values.numel() == 0:
            return 0.0
        return float(values.detach().mean().item())

    @staticmethod
    def _tensor_std(values: torch.Tensor) -> float:
        """Return the population std as a float (0.0 for fewer than 2 samples)."""
        if values.numel() < 2:
            return 0.0
        return float(values.detach().std(unbiased=False).item())

    @staticmethod
    def _explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Return ``1 - Var(target - prediction) / Var(target)``.

        Standard value-function quality measure. ``<= 0`` means the critic is no
        better than predicting the batch mean, which leaves the advantages close
        to noise and starves the policy gradient.
        """
        if target.numel() < 2:
            return 0.0
        target_variance = target.detach().var(unbiased=False)
        if float(target_variance.item()) <= 0.0:
            return 0.0
        residual = target.detach() - prediction.detach()
        return float((1.0 - residual.var(unbiased=False) / target_variance).item())

    @staticmethod
    def _action_scores(policy_state, output, action):
        """Return ``(log_prob of the taken action, entropy over feasible actions)``."""
        action_index = policy_state["actions"].index(action)
        probabilities = output["action_distribution"]
        log_prob = torch.log(probabilities[action_index].clamp_min(1e-12))
        valid_probabilities = probabilities[output["mask"]]
        entropy = -(
            valid_probabilities * torch.log(valid_probabilities.clamp_min(1e-12))
        ).sum()
        return log_prob, entropy

    def _per_transition_minibatch_rows(self, transitions, indices):
        """Recompute log-probs, entropy, and values with one forward per transition."""
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
            log_prob, entropy = self._action_scores(
                policy_state, output, transition.action
            )
            log_probs.append(log_prob)
            entropy_rows.append(entropy)
            values.append(self.value_from_output(output).reshape(()))
        return (
            torch.stack(log_probs),
            torch.stack(entropy_rows).mean(),
            torch.stack(values),
        )

    def _batched_minibatch_rows(self, transitions, indices):
        """Recompute the same quantities with one batched encoder pass.

        The GNN runs once for the whole minibatch; decoding stays per transition
        because each transition has its own candidate-action set. The critic is
        also evaluated once for the minibatch.
        """
        # DDP does not proxy custom methods, so reach the wrapped policy.
        actor = self.actor.module if hasattr(self.actor, "module") else self.actor
        contexts = [self._transition_context(transitions[index]) for index in indices]
        outputs = actor.forward_batched(contexts)

        log_probs, entropy_rows = [], []
        for index, output in zip(indices, outputs):
            log_prob, entropy = self._action_scores(
                transitions[index].state, output, transitions[index].action
            )
            log_probs.append(log_prob)
            entropy_rows.append(entropy)
        values = self.critic(
            torch.stack([self._critic_features(o) for o in outputs]).detach()
        ).squeeze(-1)
        return (
            torch.stack(log_probs),
            torch.stack(entropy_rows).mean(),
            values,
        )

    @staticmethod
    def _transition_context(transition) -> dict:
        """Extract the policy inputs stored on a rollout transition."""
        state = transition.state
        return {
            "observation": state["observation"],
            "graph": state["graph"],
            "actions": state["actions"],
            "mask": state["mask"],
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save actor, critic, both optimizers, update count, and warm-start flag."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_count": self.update_count,
            "value_bias_initialized": self._value_bias_initialized,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path, *, map_location=None) -> int:
        """Restore a checkpoint and return the update index it was saved at.

        The index lets a trainer resume from the next update instead of
        restarting the run.
        """
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        # Default to True for pre-existing checkpoints: re-running the warm start
        # on a trained critic would clobber its learned output bias.
        self._value_bias_initialized = bool(
            checkpoint.get("value_bias_initialized", True)
        )
        return int(checkpoint.get("update_idx", self.update_count))
