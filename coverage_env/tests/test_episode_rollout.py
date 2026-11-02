"""Bounded tests for complete-episode collection and replayable evaluation."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import FixedInstancePool, write_bank
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import (
    _nn_choice, run_episode, replay_recorded,
)
from coverage_env.rl.ppo.loss import compute_advantage
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.episode_rollout import EpisodeRolloutTrainer
from coverage_env.tests.test_training_runtime import ToyPolicy, toy_config


class EpisodeRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.config["environment"]["num_fields"] = {"min": 2, "max": 2}
        cls.config["field_generator"]["layout"] = {"mode": "scattered"}
        cls.config["field_generator"]["field_size"] = {"min_area": 500.0, "max_area": 800.0}
        cls.config["ppo"].update(epochs=1, batch_size=2, value_gate=True,
                                 randomize_uav=False, gamma=1.0)
        cls.config["rl"]["reward"]["mode"] = "time"
        cls.config["rl"]["decoder"]["candidate_attention"] = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        envs = [InstanceGenerator(self.config).generate(
            num_fields=2, seed=70000 + i, randomize_each_reset=False,
            randomize_uav=False) for i in range(2)]
        self.bank = self.root / "bank"
        write_bank(self.bank, self.config, envs, [70000, 70001])

    def test_complete_episodes_do_not_reset_at_batch_boundary(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=None, config=self.config,
                                        episodes_per_rank=2, max_episode_steps=200)
        trainer.train(1, rollout_steps=1)
        # Two episodes, two starts, and no zero-step instance created at the end.
        self.assertEqual(sum(pool.starts), 2)
        self.assertEqual(trainer._stats_episodes, 2)
        self.assertGreater(trainer._rank_stats["transitions"], 1)
        self.assertFalse(trainer._episode_open)
        self.assertTrue(pool._episode_done)

    def test_next_update_starts_a_new_episode_without_stale_reset(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=None, config=self.config,
                                        episodes_per_rank=1, max_episode_steps=200)
        trainer.train(2, rollout_steps=1)
        self.assertEqual(sum(pool.starts), 2)
        self.assertEqual(trainer._stats_episodes, 2)
        self.assertEqual(pool.starts, pool.completions)

    def test_episode_gae_is_isolated_and_matches_hand_calculation(self):
        # Two independent two-step episodes; the second episode's return must not
        # propagate into the first.
        reward = [1.0, 1.0, 10.0, 10.0]
        value = [0.0, 0.0, 0.0, 0.0]
        next_value = [0.0, 0.0, 0.0, 0.0]
        done = [False, True, False, True]
        adv, ret = compute_advantage(reward, value, next_value, done,
                                     gamma=1.0, gae_lambda=0.5, device="cpu")
        self.assertAlmostEqual(float(adv[1]), 1.0, places=6)
        self.assertAlmostEqual(float(adv[0]), 1.5, places=6)
        self.assertAlmostEqual(float(adv[3]), 10.0, places=6)
        self.assertAlmostEqual(float(adv[2]), 15.0, places=6)

    def test_masked_ppo_uses_per_minibatch_mean_not_rollout_scale(self):
        from types import SimpleNamespace
        agent = PPOAgent(ToyPolicy(), toy_config())
        agent.batch_size = 3
        agent.epochs = 1
        rows = [SimpleNamespace(log_prob=0.0) for _ in range(6)]

        def fake_rows(transitions, indices):
            count = len(indices)
            p = torch.zeros(1, requires_grad=True)
            return p * 0 + torch.zeros(count), torch.zeros(count), p * 0 + torch.zeros(count)
        agent._per_transition_minibatch_rows = fake_rows
        stats = agent.update_masked(rows, [True] * 6, global_valid_count=6,
                                    global_steps=6,
                                    advantages=torch.ones(6), returns=torch.zeros(6))
        self.assertEqual(stats["updates"], 2)
        # Two 3-sample minibatches each have mean policy loss 1; the old
        # rollout-wide /6 scaling would report -0.5 per minibatch.
        self.assertAlmostEqual(stats["actor_loss"], -1.0, places=6)

    def test_global_padding_only_minibatch_skips_optimizer_steps(self):
        from types import SimpleNamespace
        agent = PPOAgent(ToyPolicy(), toy_config())
        agent.batch_size = 3
        agent.epochs = 1
        rows = [SimpleNamespace(log_prob=0.0) for _ in range(6)]
        before = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}

        def fake_rows(transitions, indices):
            count = len(indices)
            p = torch.zeros(1, requires_grad=True)
            return p * 0 + torch.zeros(count), torch.zeros(count), p * 0 + torch.zeros(count)
        agent._per_transition_minibatch_rows = fake_rows
        stats = agent.update_masked(rows, [False] * 6, global_valid_count=6,
                                    global_steps=6,
                                    advantages=torch.zeros(6), returns=torch.zeros(6))
        self.assertEqual(stats["updates"], 0)
        for key, value in agent.actor.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_masked_update_ignores_padding_and_keeps_shapes_valid(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=None, config=self.config,
                                        episodes_per_rank=1, max_episode_steps=200)
        trainer._collect_episodes()
        transitions = list(trainer.buffer.transitions)
        n = len(transitions)
        self.assertGreater(n, 0)
        adv = torch.zeros(n); ret = torch.zeros(n)
        stats = agent.update_masked(transitions, [True] * n,
                                    global_valid_count=n, global_steps=n + 3,
                                    advantages=adv, returns=ret)
        self.assertEqual(stats["actor_samples"], n)
        self.assertEqual(stats["excluded_actor_samples"], 3)
        self.assertTrue(all(math.isfinite(float(stats[k])) for k in
                            ("actor_loss", "critic_loss", "entropy", "policy_loss")))

    def test_truncation_keeps_bootstrap_and_breaks_only_next_gae(self):
        # A truncated last step boots V(next), but GAE must not leak past it.
        adv, _ret = compute_advantage([1.0, 1.0], [0.0, 0.0], [0.0, 2.0],
                                      [False, False], gamma=1.0, gae_lambda=0.5,
                                      truncated=[False, True], device="cpu")
        self.assertAlmostEqual(float(adv[1]), 3.0, places=6)
        self.assertAlmostEqual(float(adv[0]), 1.0 + 0.5 * 3.0, places=6)

    def test_sampling_failure_is_synchronized_and_diagnosed(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        out = self.root / "failure_collect"
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=out, checkpoint_every=1,
                                        config=self.config, episodes_per_rank=1,
                                        max_episode_steps=200)
        trainer._collect_episodes = lambda: (_ for _ in ()).throw(RuntimeError("injected collect failure"))
        with self.assertRaises(RuntimeError):
            trainer.train(1, rollout_steps=1)
        self.assertTrue((out / "run_failures/update_1_collect.json").exists())

    def test_gate_construction_failure_is_synchronized_and_diagnosed(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        out = self.root / "failure_gate"
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=out, checkpoint_every=1,
                                        config=self.config, episodes_per_rank=1,
                                        max_episode_steps=200)
        trainer._local_gate_metrics = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected gate failure"))
        with self.assertRaises(RuntimeError):
            trainer.train(1, rollout_steps=1)
        self.assertTrue((out / "run_failures/update_1_gate_construct.json").exists())

    def test_recorded_action_replay_is_independent(self):
        manifest = json.loads((self.bank / "manifest.json").read_text())
        env = FixedInstancePool(self.bank, seed=42)
        row = run_episode(env, _nn_choice, record=True)
        self.assertTrue(row["completed"])
        self.assertTrue(row["actions"])
        repaired = replay_recorded(env._active, row["actions"])
        self.assertTrue(repaired["completed"])
        self.assertAlmostEqual(repaired["total_time"], row["total_time"], places=6)

    def test_failed_episode_replay_is_compared_not_forced_to_success(self):
        env = FixedInstancePool(self.bank, seed=42)
        row = run_episode(env, lambda _e, _s: (None, 0.0, 0.0), record=True)
        self.assertFalse(row["completed"])
        self.assertEqual(row["termination_reason"], "no_action")
        replay = replay_recorded(env._active, row["actions"])
        self.assertFalse(replay["completed"])
        self.assertEqual(replay["termination_reason"], row["termination_reason"])
        self.assertAlmostEqual(replay["total_time"], row["total_time"], places=6)

    def test_eval_actions_are_saved_when_checkpoint_directory_is_used(self):
        pool = FixedInstancePool(self.bank, seed=42)
        agent = PPOAgent(CoveragePolicy(pool, config=self.config), self.config, device="cpu")
        out = self.root / "run"
        trainer = EpisodeRolloutTrainer(pool, agent, rank=0, world_size=1,
                                        checkpoint_dir=out, checkpoint_every=1,
                                        config=self.config, episodes_per_rank=1,
                                        max_episode_steps=200)
        trainer.train(1, rollout_steps=1)
        data = json.loads((out / "evaluation/update_0.json").read_text())
        row = data["rows"][0]
        self.assertIn("actions", row)
        self.assertIn("state_summaries", row)
        self.assertTrue(row["replay_ok"])
        self.assertTrue(row["completed"])


if __name__ == "__main__":
    unittest.main()
