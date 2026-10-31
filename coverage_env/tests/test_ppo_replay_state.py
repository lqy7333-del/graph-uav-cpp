"""PPO replay must condition on the transition's stored state, not the live env.

Regression test for a correctness bug: ``CoveragePolicy._global_sources`` used
to prefer ``self.environment`` (the trainer's live environment) over the
observation stored with the transition. By the time ``PPOAgent.update()`` runs,
that environment has advanced past the rollout, so the replayed ``z_G`` — and
therefore the log-probs and values — disagreed with the ones recorded during
collection. That breaks the on-policy assumption (the PPO ratio is not 1 at the
first epoch) and silently corrupts the clipped objective.
"""

from __future__ import annotations

import unittest
from copy import deepcopy

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.global_state_encoder import (
    build_depot_feature,
    build_uav_feature,
)
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.trainer import _clone_graph


def make_config(*, epochs: int = 2, batch_size: int = 4) -> dict:
    """Return a small config with dropout off so replay is deterministic."""
    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 5, "max": 10}
    config["ppo"].update({"epochs": epochs, "batch_size": batch_size})
    config["rl"]["encoder"]["dropout"] = 0.0
    return config


def collect_rollout(actor: CoveragePolicy, environment, steps: int = 10) -> RolloutBuffer:
    """Collect with ``actor`` itself so the stored log-probs can be replayed."""
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
    return buffer


def make_environment_stale(environment) -> None:
    """Move the live environment far away from where the rollout ended."""
    uav = environment.uavs[0]
    uav.current_position = (9999.0, -9999.0)
    uav.remaining_energy = 1.0
    uav.remaining_payload = 0.0


class PpoReplayStateTests(unittest.TestCase):
    """The replayed global context must come from the stored observation."""

    @classmethod
    def setUpClass(cls):
        cls.config = make_config()
        cls.environment = InstanceGenerator(cls.config).generate(seed=20261015)
        cls.actor = CoveragePolicy(cls.environment, config=cls.config)
        cls.actor.eval()
        cls.buffer = collect_rollout(cls.actor, cls.environment)
        cls.transitions = list(cls.buffer.transitions)

    def test_uav_and_depot_features_agree_between_object_and_state(self):
        """The fix is a no-op during rollout: both sources give the same vector."""
        environment = InstanceGenerator(self.config).generate(seed=11)
        state = environment.reset()
        uav = environment.uavs[0]
        self.assertTrue(torch.equal(
            build_uav_feature(uav),
            build_uav_feature(state["uavs"][uav.uav_id]),
        ))
        self.assertTrue(torch.equal(
            build_depot_feature(environment.depot),
            build_depot_feature(state["depot"]),
        ))

    def test_global_sources_prefers_the_stored_state(self):
        observation = self.transitions[0].state["observation"]
        uav_id = self.environment.uavs[0].uav_id
        with self.subTest("state present"):
            uav_source, depot_source = CoveragePolicy._global_sources(
                self.environment, observation
            )
            self.assertIs(uav_source, observation["uavs"][uav_id])
            self.assertIs(depot_source, observation["depot"])
        with self.subTest("no state falls back to environment"):
            uav_source, depot_source = CoveragePolicy._global_sources(
                self.environment, None
            )
            self.assertIs(uav_source, self.environment.uavs[0])
            self.assertIs(depot_source, self.environment.depot)
        with self.subTest("state without uavs falls back to environment"):
            uav_source, _ = CoveragePolicy._global_sources(
                self.environment, {"depot": observation["depot"]}
            )
            self.assertIs(uav_source, self.environment.uavs[0])

    def test_replay_ignores_a_stale_environment(self):
        """Mutating the live environment must not change a replayed embedding."""
        policy = deepcopy(self.actor).eval()
        transition = self.transitions[0]
        context = {
            "graph": transition.state["graph"],
            "candidate_actions": transition.state["actions"],
            "feasible_mask": transition.state["mask"],
        }
        observation = transition.state["observation"]
        with torch.no_grad():
            baseline = policy(
                observation, **context
            )["global_embedding"]

        make_environment_stale(policy.environment)
        stale = build_uav_feature(policy.environment.uavs[0])
        stored = build_uav_feature(
            observation["uavs"][policy.environment.uavs[0].uav_id]
        )
        self.assertFalse(torch.allclose(stale, stored), "test setup: env not stale")

        with torch.no_grad():
            after = policy(observation, **context)["global_embedding"]
        self.assertTrue(torch.equal(baseline, after))

    def test_replay_reproduces_stored_log_probs(self):
        """Recomputing log-probs must reproduce collection exactly (ratio == 1)."""
        policy = deepcopy(self.actor).eval()
        make_environment_stale(policy.environment)
        agent = PPOAgent(policy, self.config)
        log_probs, _entropy, _values = agent._per_transition_minibatch_rows(
            self.transitions, list(range(len(self.transitions)))
        )
        stored = torch.tensor([row.log_prob for row in self.transitions])
        self.assertTrue(torch.allclose(log_probs.detach(), stored, atol=1e-5))

    def test_batched_replay_reproduces_stored_log_probs(self):
        policy = deepcopy(self.actor).eval()
        make_environment_stale(policy.environment)
        agent = PPOAgent(policy, self.config)
        log_probs, _entropy, _values = agent._batched_minibatch_rows(
            self.transitions, list(range(len(self.transitions)))
        )
        stored = torch.tensor([row.log_prob for row in self.transitions])
        self.assertTrue(torch.allclose(log_probs.detach(), stored, atol=1e-5))

    def test_bootstrap_value_uses_the_next_observation_heading(self):
        """V(s_{t+1}) must use the stored heading, not the env's missing one."""
        environment = InstanceGenerator(self.config).generate(seed=20261016)
        actor = CoveragePolicy(environment, config=self.config).eval()
        agent = PPOAgent(actor, self.config)
        state = environment.reset()
        with torch.no_grad():
            output = actor(state, environment=environment)
            action = CoveragePolicy.greedy_action(output)
        state, _reward, _done, _info = environment.step(action)
        uav_id = environment.uavs[0].uav_id
        self.assertTrue(state["uavs"][uav_id]["heading_valid"])
        graph = actor.graph_builder.build_graph(environment)
        with torch.no_grad():
            value_state = float(agent.value_from_output(
                actor(state, environment=environment)).item())
            value_graph_state = agent.value_for_graph(
                graph, environment, state=state)
            value_graph_env = agent.value_for_graph(graph, environment)
        self.assertAlmostEqual(value_state, value_graph_state, places=5)
        self.assertGreater(abs(value_state - value_graph_env), 1e-3)

    def test_bootstrap_value_after_resupply_has_no_heading(self):
        """After a depot stop both state and env must agree on heading_valid=False."""
        environment = InstanceGenerator(self.config).generate(seed=20261017)
        actor = CoveragePolicy(environment, config=self.config).eval()
        agent = PPOAgent(actor, self.config)
        environment.reset()
        environment.return_to_depot()
        state = environment.get_state()
        uav_id = environment.uavs[0].uav_id
        self.assertFalse(state["uavs"][uav_id]["heading_valid"])
        graph = actor.graph_builder.build_graph(environment)
        with torch.no_grad():
            value_state = float(agent.value_from_output(
                actor(state, environment=environment)).item())
            value_graph_state = agent.value_for_graph(graph, environment, state=state)
            value_graph_env = agent.value_for_graph(graph, environment)
        self.assertAlmostEqual(value_state, value_graph_state, places=5)
        self.assertAlmostEqual(value_state, value_graph_env, places=5)

    def test_rollout_tail_bootstrap_uses_next_state(self):
        """The transition flagged truncated must bootstrap the real next state."""
        environment = InstanceGenerator(self.config).generate(seed=20261018)
        actor = CoveragePolicy(environment, config=self.config).eval()
        agent = PPOAgent(actor, self.config)
        state = environment.reset()
        with torch.no_grad():
            output = actor(state, environment=environment)
            action = CoveragePolicy.greedy_action(output)
        next_state, _reward, done, _info = environment.step(action)
        if done:
            self.skipTest("seed completed in one step")
        graph = actor.graph_builder.build_graph(environment)
        with torch.no_grad():
            value_state = float(agent.value_from_output(
                actor(next_state, environment=environment)).item())
            value_bootstrap = agent.value_for_graph(graph, environment, state=next_state)
        self.assertAlmostEqual(value_state, value_bootstrap, places=5)

    def test_terminal_and_truncated_gae_boundaries(self):
        from coverage_env.rl.ppo.loss import compute_advantage
        # Terminal: next_value=999 must be ignored for delta and return.
        adv, ret = compute_advantage(
            [-2.0], [-1.0], [999.0], [True], 1.0, 0.95,
            device="cpu", truncated=[False])
        self.assertAlmostEqual(float(adv[0]), -1.0, places=6)
        self.assertAlmostEqual(float(ret[0]), -2.0, places=6)
        # Truncated tail: bootstrap V(next)=-3, but do not leak GAE across boundary.
        adv, ret = compute_advantage(
            [-2.0], [-1.0], [-3.0], [False], 1.0, 0.95,
            device="cpu", truncated=[True])
        self.assertAlmostEqual(float(adv[0]), -4.0, places=6)
        self.assertAlmostEqual(float(ret[0]), -5.0, places=6)

    def test_rollout_encoding_is_unchanged(self):
        """While the environment is current, both sources encode identically."""
        environment = InstanceGenerator(self.config).generate(seed=11)
        actor = CoveragePolicy(environment, config=self.config)
        actor.eval()
        state = environment.reset()
        graph = actor.graph_builder.build_graph(environment)
        with torch.no_grad():
            with_environment = actor.encode_graph(graph, environment=environment)[1]
            with_state = actor.encode_graph(graph, state=state)[1]
        self.assertTrue(torch.allclose(with_environment, with_state, atol=1e-6))


if __name__ == "__main__":
    import sys

    configuration = make_config()
    environment = InstanceGenerator(configuration).generate(seed=20261015)
    actor = CoveragePolicy(environment, config=configuration)
    actor.eval()
    buffer = collect_rollout(actor, environment)
    transitions = list(buffer.transitions)
    print(f"collected {len(transitions)} transitions")

    transitions = list(buffer.transitions)
    stored = torch.tensor([row.log_prob for row in transitions])
    for label, batched in (("per-transition", False), ("batched", True)):
        policy = deepcopy(actor).eval()
        make_environment_stale(policy.environment)
        probe = PPOAgent(policy, configuration)
        rows = (probe._batched_minibatch_rows if batched
                else probe._per_transition_minibatch_rows)
        log_probs, _entropy, _values = rows(transitions, list(range(len(transitions))))
        difference = float((log_probs.detach() - stored).abs().max().item())
        print(f"  {label:>14}: max |replayed - stored| log-prob = {difference:.3e}")
    sys.exit(0)
