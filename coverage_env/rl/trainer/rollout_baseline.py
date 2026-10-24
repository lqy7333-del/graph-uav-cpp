"""REINFORCE with a frozen greedy rollout baseline.

Why this exists: PPO's critic gives a noisy advantage when the return is dominated
by a long, near-constant distance term, so the actor can plateau while the critic
explains the variance (EV is high, advantage is not). The alternative is the classic
routing recipe from Kool et al.: sample one trajectory, roll out a frozen greedy
baseline *on the same instance*, and use the paired cost difference as the advantage.

Scope of this module: it only implements the training method. The GNN, action
embedding, hierarchical decoder, action semantics, workload decisions, automatic
resupply, executor, environment dynamics, reward, gamma (asserted to be 1.0) and the
instance distribution are untouched.
"""
from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field

import torch

from coverage_env.env.instance_generator import InstanceGenerator


@dataclass
class EpisodeResult:
    """One sampled trajectory and the matching baseline rollout on the same instance."""

    seed: int
    success: bool
    sample_cost: float
    baseline_cost: float | None
    log_prob_sum: torch.Tensor
    steps: int
    termination_reason: str

    @property
    def advantage(self) -> float:
        """Positive when the sampled trajectory is cheaper than the baseline."""
        if self.baseline_cost is None:
            return 0.0
        return self.baseline_cost - self.sample_cost


@dataclass
class RolloutBaselineStats:
    episodes: int = 0
    successes: int = 0
    baseline_rollouts: int = 0
    baseline_rollout_steps: int = 0
    promotions: int = 0
    promotion_log: list = field(default_factory=list)


def _assert_unit_gamma(ppo_config):
    gamma = float(ppo_config.get("gamma", 0.99))
    if abs(gamma - 1.0) > 1e-9:
        raise ValueError(
            f"rollout-baseline training requires gamma == 1.0 (got {gamma}). "
            "Passing the code default 0.99 here would silently change the method."
        )


def _build_instance(config, num_fields, seed):
    return InstanceGenerator(config).generate(
        num_fields=num_fields, seed=seed, randomize_each_reset=False
    )


def _sample_episode(environment, policy, *, seed):
    """Sample a full mission; return (cost, log_prob_sum, steps, success, reason)."""
    torch.manual_seed(seed)
    environment.reset()
    log_prob_sum = torch.zeros(())
    total_reward = 0.0
    steps = 0
    done = False
    info: dict = {}
    policy.eval()
    while not done and steps < 500:
        state = environment.get_state()
        output = policy(state, environment=environment)
        distribution = output["action_distribution"]
        if not torch.isfinite(distribution).all() or float(distribution.sum()) <= 0.0:
            break
        index = int(torch.multinomial(distribution.detach().cpu(), 1).item())
        action = output["actions"][index]
        # The policy exposes the distribution, not precomputed log-probs.
        log_prob_sum = log_prob_sum + torch.log(distribution[index].clamp_min(1e-12))
        _s, reward, done, info = environment.step(action)
        total_reward += float(reward)
        steps += 1
    success = str(info.get("termination_reason", "")) == "mission_complete"
    return -total_reward, log_prob_sum, steps, success, str(info.get("termination_reason", ""))


def _greedy_episode(environment, policy):
    """Deterministic greedy rollout; returns (cost, steps, success)."""
    environment.reset()
    total_reward = 0.0
    steps = 0
    done = False
    info: dict = {}
    policy.eval()
    with torch.no_grad():
        while not done and steps < 500:
            output = policy(environment.get_state(), environment=environment)
            action = policy.greedy_action(output)
            _s, reward, done, info = environment.step(action)
            total_reward += float(reward)
            steps += 1
    success = str(info.get("termination_reason", "")) == "mission_complete"
    return -total_reward, steps, success


class RolloutBaselineTrainer:
    """REINFORCE updates against a frozen greedy baseline on the same instances."""

    def __init__(self, agent, config, *, instance_seeds, num_fields=(5, 20),
                 learning_rate=None, promotion_instances=20, promotion_alpha=0.10):
        # Accept either the policy itself or an agent that wraps one: PPOAgent
        # exposes the network under a different attribute, so requiring ``.policy``
        # here made the trainer unusable with the real agent.
        self.agent = agent
        self.policy = getattr(agent, "policy", None) or getattr(agent, "actor", agent)
        self.config = config
        _assert_unit_gamma(config["ppo"])
        self.baseline = copy.deepcopy(self.policy)
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(learning_rate if learning_rate is not None
                     else config["ppo"].get("learning_rate", 3e-4)),
        )
        self.instance_seeds = list(instance_seeds)
        self.num_fields = num_fields
        self.promotion_instances = int(promotion_instances)
        self.promotion_alpha = float(promotion_alpha)
        self.stats = RolloutBaselineStats()

    # -- training ---------------------------------------------------------
    def _episode(self, seed, num_fields):
        sample_env = _build_instance(self.config, num_fields, seed)
        sample_cost, log_prob_sum, steps, success, reason = _sample_episode(
            sample_env, self.policy, seed=seed
        )
        baseline_cost = None
        if success:
            baseline_env = _build_instance(self.config, num_fields, seed)
            baseline_cost, baseline_steps, baseline_success = _greedy_episode(
                baseline_env, self.baseline
            )
            self.stats.baseline_rollouts += 1
            self.stats.baseline_rollout_steps += baseline_steps
            if not baseline_success:
                baseline_cost = None
        return EpisodeResult(seed=seed, success=success, sample_cost=sample_cost,
                             baseline_cost=baseline_cost, log_prob_sum=log_prob_sum,
                             steps=steps, termination_reason=reason)

    def update(self, *, num_episodes=8):
        """One optimizer step over ``num_episodes`` sampled episodes."""
        import random
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch_loss = None
        used = 0
        for seed in random.Random(self.stats.episodes).sample(
            self.instance_seeds, min(num_episodes, len(self.instance_seeds))
        ):
            num_fields = random.Random(seed).choice(
                list(range(self.num_fields[0], self.num_fields[1] + 1))
            )
            result = self._episode(seed, num_fields)
            self.stats.episodes += 1
            if not result.success or result.baseline_cost is None:
                continue
            self.stats.successes += 1
            advantage = result.advantage
            term = -advantage * result.log_prob_sum / max(num_episodes, 1)
            batch_loss = term if batch_loss is None else batch_loss + term
            used += 1
        if batch_loss is None:
            return {"loss": 0.0, "episodes": used, "skipped": num_episodes}
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": float(batch_loss.detach()), "episodes": used, "skipped": num_episodes - used}

    # -- baseline promotion ----------------------------------------------
    def promotion_check(self, promotion_seeds):
        """Greedy paired comparison on promotion instances; promote only if significant."""
        current_wins, deltas = 0, []
        for index, seed in enumerate(promotion_seeds[: self.promotion_instances]):
            num_fields = 5 + (index % 4) * 5
            env_current = _build_instance(self.config, num_fields, seed)
            env_baseline = _build_instance(self.config, num_fields, seed)
            current_cost, _cs, current_ok = _greedy_episode(env_current, self.policy)
            baseline_cost, _bs, baseline_ok = _greedy_episode(env_baseline, self.baseline)
            if not (current_ok and baseline_ok):
                continue
            deltas.append(baseline_cost - current_cost)
            current_wins += int(deltas[-1] > 0.0)
        if len(deltas) < 5:
            return {"promoted": False, "reason": "insufficient promotion samples",
                    "n": len(deltas)}
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        standard_error = sd / (len(deltas) ** 0.5) if sd > 0 else 0.0
        t = mean / standard_error if standard_error > 0 else 0.0
        # Two-sided 95% interval on the paired mean (t critical ~2.1 for n>=13).
        ci_half = 2.1 * standard_error
        # One-sided: promote only when the current actor is better by a margin.
        promoted = t > 2.0 and mean > 0.0
        if promoted:
            reason = "promoted: paired mean improvement is significant"
        elif mean <= 0.0:
            reason = "blocked: paired mean improvement is not positive"
        elif t <= 2.0:
            reason = "blocked: t statistic below the 2.0 threshold"
        else:
            reason = "blocked: unknown"
        record = {"n": len(deltas), "mean_delta": mean, "sd": sd, "t": t,
                  "ci95": [mean - ci_half, mean + ci_half],
                  "wins": current_wins, "promoted": promoted, "reason": reason}
        if promoted:
            self.baseline.load_state_dict(self.policy.state_dict())
            for parameter in self.baseline.parameters():
                parameter.requires_grad_(False)
            self.stats.promotions += 1
        self.stats.promotion_log.append(record)
        return record
