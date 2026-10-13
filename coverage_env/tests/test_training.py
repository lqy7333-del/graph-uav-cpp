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
        actor_before = copy.deepcopy(actor.action_decoder.network[-1].weight.detach())
        critic_before = copy.deepcopy(agent.critic.network[-1].weight.detach())

        history = trainer.train(num_episode=10)
        self.assertEqual(len(history), 10)
        self.assertGreater(trainer.last_rollout_size, 0)
        self.assertEqual(len(trainer.buffer), 0)
        self.assertIn("actor_loss", history[-1])
        self.assertIn("critic_loss", history[-1])
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["actor_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["critic_loss"])))
        self.assertFalse(torch.equal(actor_before, actor.action_decoder.network[-1].weight))
        self.assertFalse(torch.equal(critic_before, agent.critic.network[-1].weight))

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            checkpoint = Path(directory) / "ppo.pt"
            agent.save_checkpoint(checkpoint)
            agent.load_checkpoint(checkpoint)

        metrics = Evaluator(environment, actor, max_steps=1).evaluate(1)
        for key in ("reward", "coverage_ratio", "total_distance", "energy_consumption", "success_rate"):
            self.assertIn(key, metrics)
        print("PPO smoke result:", history[-1], "evaluation:", metrics)


if __name__ == "__main__":
    unittest.main()
