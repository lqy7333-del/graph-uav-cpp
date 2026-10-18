"""Short PPO smoke test covering rollout storage and parameter updates."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.evaluator import Evaluator
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.trainer import Trainer


class PPOTrainingTests(unittest.TestCase):
    """Run ten short episodes and verify buffer, optimization, and checkpoint."""

    def test_ten_episode_training_smoke(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["environment"]["num_fields"] = {"min": 1, "max": 1}
        config["ppo"].update({"epochs": 2, "batch_size": 8, "learning_rate": 1e-3})
        environment = InstanceGenerator(config).generate(num_fields=1, seed=712)
        actor = CoveragePolicy(environment, config=config)
        agent = PPOAgent(actor, config)
        trainer = Trainer(environment, agent, config=config, max_steps=1)
        # Checksum the whole actor rather than one head: this test runs a
        # single-parcel instance, where the parcel level of the factored decode is
        # degenerate (one feasible parcel => p=1 => exactly zero gradient), so a
        # head-specific probe would be testing the wrong thing.
        actor_before = torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
        critic_before = copy.deepcopy(agent.critic.network[-1].weight.detach())

        history = trainer.train(num_episode=10)
        self.assertEqual(len(history), 10)
        self.assertGreater(trainer.last_rollout_size, 0)
        self.assertEqual(len(trainer.buffer), 0)
        self.assertIn("actor_loss", history[-1])
        self.assertIn("critic_loss", history[-1])
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["actor_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["critic_loss"])))
        actor_after = torch.cat([p.detach().flatten() for p in actor.parameters()])
        self.assertFalse(torch.allclose(actor_before, actor_after))
        self.assertFalse(torch.equal(critic_before, agent.critic.network[-1].weight))

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            checkpoint = Path(directory) / "ppo.pt"
            agent.save_checkpoint(checkpoint)
            agent.load_checkpoint(checkpoint)

        metrics = Evaluator(environment, actor, max_steps=1).evaluate(1)
        for key in ("reward", "coverage_ratio", "total_distance", "energy_consumption", "success_rate"):
            self.assertIn(key, metrics)
        print("PPO smoke result:", history[-1], "evaluation:", metrics)


class ValueWarmStartTests(unittest.TestCase):
    """The critic starts at the return mean instead of crawling up to it."""

    def test_bias_is_set_from_the_first_return_batch_once(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["environment"]["num_fields"] = {"min": 5, "max": 5}
        environment = InstanceGenerator(config).generate(num_fields=5, seed=3)
        actor = CoveragePolicy(environment, config=config)
        agent = PPOAgent(actor, config)
        module = agent.critic.module if hasattr(agent.critic, "module") else agent.critic

        self.assertFalse(agent._value_bias_initialized)
        # nn.Linear initialises the bias uniformly, not to zero; the point is
        # that it does not yet sit at the return mean.
        self.assertNotAlmostEqual(
            float(module.network[-1].bias.item()), -18.0, places=3
        )

        agent._warm_start_value_bias(torch.tensor([-18.0, -20.0, -16.0]))
        self.assertTrue(agent._value_bias_initialized)
        self.assertAlmostEqual(float(module.network[-1].bias.item()), -18.0, places=5)

        # One-shot: later calls must not clobber the learned bias.
        agent._warm_start_value_bias(torch.tensor([-99.0]))
        self.assertAlmostEqual(float(module.network[-1].bias.item()), -18.0, places=5)


class LearningRateTests(unittest.TestCase):
    """Adam's step is ~lr, so each network needs a rate sized to its own range."""

    def _agent(self, config: dict):
        config["environment"]["num_fields"] = {"min": 5, "max": 5}
        environment = InstanceGenerator(config).generate(num_fields=5, seed=3)
        return PPOAgent(CoveragePolicy(environment, config=config), config)

    def test_default_config_keeps_the_actor_at_the_base_rate(self):
        """Measured: a 10x actor rate made the greedy policy worse, so it stays base."""
        agent = self._agent(load_config(DEFAULT_CONFIG_PATH))
        self.assertAlmostEqual(agent.actor_learning_rate, 0.0003, places=8)
        self.assertAlmostEqual(
            agent.actor_optimizer.param_groups[0]["lr"], 0.0003, places=8
        )
        # The critic's 10x, by contrast, is measured-good and kept.
        self.assertAlmostEqual(
            agent.critic_optimizer.param_groups[0]["lr"], 0.003, places=8
        )

    def test_rates_can_be_overridden_independently(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["ppo"].update(
            {"actor_learning_rate": 0.02, "critic_learning_rate": 0.05}
        )
        agent = self._agent(config)
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 0.02)
        self.assertAlmostEqual(agent.critic_optimizer.param_groups[0]["lr"], 0.05)

    def test_missing_keys_fall_back_to_the_shared_rate(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["ppo"].pop("actor_learning_rate", None)
        config["ppo"].pop("critic_learning_rate", None)
        agent = self._agent(config)
        self.assertAlmostEqual(agent.actor_optimizer.param_groups[0]["lr"], 0.0003)
        self.assertAlmostEqual(agent.critic_optimizer.param_groups[0]["lr"], 0.0003)


class CheckpointResumeTests(unittest.TestCase):
    """Checkpoints must carry enough state to continue a run."""

    def _agent(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["environment"]["num_fields"] = {"min": 5, "max": 5}
        environment = InstanceGenerator(config).generate(num_fields=5, seed=3)
        return PPOAgent(CoveragePolicy(environment, config=config), config)

    def _critic_bias(self, agent):
        module = agent.critic.module if hasattr(agent.critic, "module") else agent.critic
        return float(module.network[-1].bias.item())

    def test_load_returns_the_update_index_and_preserves_warm_start(self):
        agent = self._agent()
        agent.update_count = 7
        agent._warm_start_value_bias(torch.tensor([-21.0]))

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "ckpt.pt"
            agent.save_checkpoint(path)

            restored = self._agent()
            self.assertFalse(restored._value_bias_initialized)
            self.assertEqual(restored.load_checkpoint(path), 7)
            self.assertEqual(restored.update_count, 7)
            # Resuming must not re-trigger the warm start and clobber the bias.
            self.assertTrue(restored._value_bias_initialized)
            self.assertAlmostEqual(self._critic_bias(restored), -21.0, places=5)

    def test_train_skips_updates_a_checkpoint_already_covers(self):
        from coverage_env.rl.trainer.distributed import DistributedTrainer

        config = load_config(DEFAULT_CONFIG_PATH)
        config["environment"]["num_fields"] = {"min": 5, "max": 5}
        environment = InstanceGenerator(config).generate(num_fields=5, seed=3)
        agent = PPOAgent(CoveragePolicy(environment, config=config), config)
        trainer = DistributedTrainer(environment, agent)
        trainer.train(3, 4, start_update=4)  # already past the target
        self.assertEqual(agent.update_count, 0)
        self.assertEqual(len(trainer.buffer), 0)


class AdvantageConfigTests(unittest.TestCase):
    """The GAE bias/variance knob must actually reach the agent."""

    def _agent(self, config: dict):
        config["environment"]["num_fields"] = {"min": 5, "max": 5}
        environment = InstanceGenerator(config).generate(num_fields=5, seed=3)
        return PPOAgent(CoveragePolicy(environment, config=config), config)

    def test_gae_lambda_override_reaches_the_agent(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertAlmostEqual(self._agent(config).gae_lambda, 0.95)
        config["ppo"]["gae_lambda"] = 0.0
        self.assertAlmostEqual(self._agent(config).gae_lambda, 0.0)

    def test_lower_lambda_shrinks_the_advantage_scale(self):
        """A lambda near 0 uses the low-variance one-step residual."""
        from coverage_env.rl.ppo.loss import compute_advantage

        rewards = torch.tensor([-3.0, -3.0, -3.0, -3.0])
        values = torch.tensor([-10.0, -8.0, -6.0, -4.0])
        next_values = torch.tensor([-8.0, -6.0, -4.0, -6.0])
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0])
        high, _ = compute_advantage(
            rewards, values, next_values, dones, 0.99, 0.95
        )
        low, _ = compute_advantage(
            rewards, values, next_values, dones, 0.99, 0.0
        )
        self.assertLess(float(low.var()), float(high.var()))


if __name__ == "__main__":
    unittest.main()


class EntropyAnnealingTests(unittest.TestCase):
    """The entropy bonus must be held, then decayed, and off when unset.

    A constant bonus left the policy at 3.41 nats of entropy after 400 updates
    (77% of the log(84)=4.43 uniform maximum), so the policy being *evaluated*
    greedily was far ahead of the policy being sampled -- and the advantage that
    ordering has to come through was dominated by exploration noise.
    """

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        # `num_fields` must agree with the configured sampling range, so build the
        # instance after widening the range to a single parcel.
        self.config["environment"]["num_fields"] = {"min": 1, "max": 1}
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=1, seed=7
        )

    def _agent(self, **overrides):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["ppo"].update(overrides)
        return PPOAgent(CoveragePolicy(self.environment, config=config), config)

    def test_schedule_holds_then_decays_to_zero(self):
        agent = self._agent(
            entropy_coef=0.01, entropy_coef_end=0.0, entropy_anneal_start=0.25
        )
        self.assertAlmostEqual(agent.set_entropy_progress(0.0), 0.01)
        self.assertAlmostEqual(agent.set_entropy_progress(0.25), 0.01)
        self.assertAlmostEqual(agent.set_entropy_progress(0.625), 0.005)
        self.assertAlmostEqual(agent.set_entropy_progress(1.0), 0.0)

    def test_disabled_when_end_is_unset(self):
        agent = self._agent(entropy_coef=0.01, entropy_coef_end=None)
        self.assertAlmostEqual(agent.set_entropy_progress(1.0), 0.01)
