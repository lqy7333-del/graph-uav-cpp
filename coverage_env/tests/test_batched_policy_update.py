"""Batched-encoder policy path: equivalence with the per-transition path.

Completes the "batch the encoder only" prototype (see
``docs/training_performance_notes.md`` P1): ``CoveragePolicy.forward_batched``
must reproduce, per transition, exactly what ``CoveragePolicy.forward`` returns,
and ``PPOAgent.update(batched=True)`` must produce the same losses as the
default per-transition update.
"""

from __future__ import annotations

import time
import unittest
from copy import deepcopy

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.trainer import _clone_graph


def make_config(*, dropout: float = 0.0, epochs: int = 2, batch_size: int = 4) -> dict:
    """Return a small, deterministic training config for equivalence checks."""
    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 5, "max": 12}
    config["ppo"].update({"epochs": epochs, "batch_size": batch_size})
    config["rl"]["encoder"]["dropout"] = dropout
    return config


def collect_transitions(config: dict, steps: int = 12, seed: int = 20261015):
    """Collect ``steps`` transitions from a live, domain-randomized rollout."""
    environment = InstanceGenerator(config).generate(seed=seed)
    actor = CoveragePolicy(environment, config=config)
    buffer = RolloutBuffer()
    state = environment.reset()
    with torch.no_grad():
        for _ in range(steps):
            output = actor(state, environment=environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0.0:
                state = environment.reset()
                continue
            index = int(torch.multinomial(distribution, 1).item())
            action = output["actions"][index]
            next_state, reward, done, _info = environment.step(action)
            next_graph = actor.graph_builder.build_graph(environment)
            buffer.add(
                {
                    "observation": deepcopy(state),
                    "graph": _clone_graph(output["graph"]),
                    "actions": list(output["actions"]),
                    "mask": output["mask"].detach().cpu().tolist(),
                },
                action,
                reward,
                {"observation": deepcopy(next_state), "graph": _clone_graph(next_graph)},
                done,
                float(torch.log(distribution[index].clamp_min(1e-12))),
                0.0,
                0.0,
            )
            state = environment.reset() if done else next_state
    return environment, actor, buffer


def policy_rows(policy, transitions, *, batched: bool) -> list[dict]:
    """Return the PPO-consumed quantities for each transition from one path."""
    contexts = [
        {
            "observation": transition.state["observation"],
            "graph": transition.state["graph"],
            "actions": transition.state["actions"],
            "mask": transition.state["mask"],
        }
        for transition in transitions
    ]
    if batched:
        outputs = policy.forward_batched(contexts)
    else:
        outputs = [
            policy(
                context["observation"],
                graph=context["graph"],
                candidate_actions=context["actions"],
                feasible_mask=context["mask"],
            )
            for context in contexts
        ]
    rows = []
    for transition, context, output in zip(transitions, contexts, outputs):
        probabilities = output["action_distribution"]
        valid = probabilities[output["mask"]]
        index = context["actions"].index(transition.action)
        rows.append({
            "mask": output["mask"],
            "distribution": probabilities.detach(),
            "global_embedding": output["global_embedding"].detach(),
            "log_prob": torch.log(probabilities[index].clamp_min(1e-12)).detach(),
            "entropy": (
                -(valid * torch.log(valid.clamp_min(1e-12))).sum().detach()
            ),
        })
    return rows


def build_agent(config: dict, device=None) -> PPOAgent:
    """Return a freshly, identically seeded agent for A/B comparisons."""
    torch.manual_seed(0)
    actor = CoveragePolicy(None, config=config)
    torch.manual_seed(0)
    return PPOAgent(actor, config, device=device)


class DdpLikeWrapper:
    """Minimal stand-in for DDP's ``.module`` indirection (no process group).

    ``update(batched=True)`` must reach the wrapped policy through ``.module``
    because DDP does not proxy custom methods.
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self


class BatchedPolicyEquivalenceTests(unittest.TestCase):
    """One batched encoder pass must match one forward per transition."""

    @classmethod
    def setUpClass(cls):
        cls.config = make_config()
        cls.environment, cls.actor, cls.buffer = collect_transitions(cls.config)

    def _compare(self, policy):
        transitions = list(self.buffer.transitions)
        policy = deepcopy(policy).eval()
        with torch.no_grad():
            single = policy_rows(policy, transitions, batched=False)
            batched = policy_rows(policy, transitions, batched=True)
        self.assertEqual(len(single), len(batched))
        worst = {"distribution": 0.0, "global_embedding": 0.0,
                 "log_prob": 0.0, "entropy": 0.0}
        for index, (expected, actual) in enumerate(zip(single, batched)):
            self.assertTrue(torch.equal(expected["mask"], actual["mask"]),
                            f"mask mismatch at transition {index}")
            for key in worst:
                worst[key] = max(
                    worst[key], float((expected[key] - actual[key]).abs().max().item())
                )
        return worst

    def test_forward_batched_matches_with_bound_environment(self):
        # Mirrors the real trainer: CoveragePolicy holds the live environment.
        worst = self._compare(self.actor)
        for key, value in worst.items():
            self.assertLessEqual(value, 1e-5, f"{key}: {worst}")

    def test_forward_batched_matches_with_state_only_policy(self):
        # No bound environment: z_G is built per transition from stored states.
        policy = CoveragePolicy(None, config=self.config)
        policy.load_state_dict(self.actor.state_dict())
        worst = self._compare(policy)
        for key, value in worst.items():
            self.assertLessEqual(value, 1e-5, f"{key}: {worst}")

    def test_batched_distributions_are_valid_probabilities(self):
        policy = deepcopy(self.actor).eval()
        with torch.no_grad():
            rows = policy_rows(policy, list(self.buffer.transitions), batched=True)
        for row in rows:
            self.assertAlmostEqual(float(row["distribution"].sum()), 1.0, places=5)
            self.assertTrue(torch.all(row["distribution"] >= 0.0))

    def test_empty_context_list_returns_nothing(self):
        self.assertEqual(self.actor.forward_batched([]), [])


class BatchedPPOUpdateTests(unittest.TestCase):
    """``update(batched=True)`` must match the default update numerically."""

    @classmethod
    def setUpClass(cls):
        cls.config = make_config()
        _environment, _actor, cls.buffer = collect_transitions(cls.config)

    def test_batched_update_matches_per_transition_update(self):
        single_agent = build_agent(self.config)
        batched_agent = build_agent(self.config)
        for first, second in zip(
            single_agent.actor.parameters(), batched_agent.actor.parameters()
        ):
            self.assertTrue(torch.equal(first, second), "agents are not identically seeded")
        single_agent.actor.eval()
        batched_agent.actor.eval()

        torch.manual_seed(1234)
        single = single_agent.update(self.buffer, batched=False)
        torch.manual_seed(1234)
        batched = batched_agent.update(self.buffer, batched=True)

        self.assertEqual(single["updates"], batched["updates"])
        # Each forward agrees to ~1e-7 (see policy_rows), but a full update
        # chains several optimizer steps, so float32 round-off is amplified.
        # Compare relatively: a real slicing/indexing bug is orders of magnitude
        # larger, so this still catches regressions without being flaky.
        for key in ("actor_loss", "critic_loss", "entropy"):
            difference = abs(single[key] - batched[key])
            tolerance = 1e-4 * max(1.0, abs(single[key]))
            self.assertLessEqual(
                difference, tolerance,
                msg=(f"{key}: single={single[key]!r} batched={batched[key]!r} "
                     f"diff={difference!r} tolerance={tolerance!r}"),
            )

    def test_batched_update_really_trains_parameters(self):
        agent = build_agent(self.config)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        stats = agent.update(self.buffer, batched=True)
        self.assertGreater(stats["updates"], 0)
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, agent.actor.parameters())
        ))

    def test_empty_buffer_is_a_no_op(self):
        agent = build_agent(self.config)
        stats = agent.update(RolloutBuffer(), batched=True)
        self.assertEqual(stats["updates"], 0)

    def test_batched_update_reaches_policy_through_ddp_wrapper(self):
        agent = build_agent(self.config)
        agent.actor = DdpLikeWrapper(agent.actor)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        stats = agent.update(self.buffer, batched=True)
        self.assertGreater(stats["updates"], 0)
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, agent.actor.parameters())
        ))


if __name__ == "__main__":
    import sys

    configuration = make_config()
    environment, actor, buffer = collect_transitions(configuration)
    transitions = list(buffer.transitions)
    field_counts = [len(t.state["actions"]) for t in transitions]
    print(f"collected {len(transitions)} transitions "
          f"({field_counts[0]}..{field_counts[-1]} candidate actions)")

    evaluator = deepcopy(actor).eval()
    with torch.no_grad():
        single = policy_rows(evaluator, transitions, batched=False)
        batched = policy_rows(evaluator, transitions, batched=True)
    print("\nper-transition vs batched policy forward (max abs diff):")
    for key in ("distribution", "global_embedding", "log_prob", "entropy"):
        diff = max(
            float((a[key] - b[key]).abs().max().item())
            for a, b in zip(single, batched)
        )
        print(f"  {key:>17}: {diff:.3e}")
    print(f"  masks identical  : {all(torch.equal(a['mask'], b['mask']) for a, b in zip(single, batched))}")

    for epochs, batch_size in ((4, 64),):
        config = make_config(epochs=epochs, batch_size=batch_size)
        _env, _actor, buf = collect_transitions(config, steps=64)
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            timings = {}
            for label, flag in (("per-transition", False), ("batched", True)):
                agent = build_agent(config, device=device)
                agent.actor.eval()
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                agent.update(buf, batched=flag)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings[label] = time.perf_counter() - start
            print(f"update({len(buf)} transitions, epochs={epochs}, "
                  f"batch={batch_size}, {device}): "
                  f"per-transition {timings['per-transition']:.2f}s | "
                  f"batched {timings['batched']:.2f}s | "
                  f"speedup {timings['per-transition']/timings['batched']:.2f}x")
    sys.exit(0)
