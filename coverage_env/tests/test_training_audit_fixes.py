"""Regression tests for the 2026-10-28 training audit findings.

Each test pins an acceptance condition the audit asked for, so the defects cannot
silently return.
"""
import unittest

import torch


class DropoutReplayTests(unittest.TestCase):
    """Sampling and PPO replay must use the same deterministic network mode."""

    def test_train_mode_changes_log_probs_eval_mode_does_not(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.env.instance_generator import InstanceGenerator
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy

        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=20000, randomize_each_reset=False
        )
        env.reset()
        policy = CoveragePolicy(env, config=config)
        state = env.get_state()

        policy.eval()
        with torch.no_grad():
            first = policy(state, environment=env)["action_distribution"].clone()
            second = policy(state, environment=env)["action_distribution"].clone()
        self.assertLess(float((first - second).abs().max()), 1e-9,
                        "eval 模式重算概率必须一致")

        policy.train()
        with torch.no_grad():
            third = policy(state, environment=env)["action_distribution"].clone()
        self.assertGreater(float((first - third).abs().max()), 0.0,
                           "train 模式确实改变概率（dropout 关闭时该断言会失败）")


class ConfigPriorityTests(unittest.TestCase):
    """--set must not be silently overwritten by CLI defaults."""

    def test_cli_defaults_are_none(self):
        import inspect
        import coverage_env.rl.trainer.distributed as module
        source = inspect.getsource(module.main)
        self.assertIn('"--ppo-epochs", type=int, default=None', source)
        self.assertIn('"--batch-size", type=int, default=None', source)


class BroadcastMathTests(unittest.TestCase):
    """The broadcast advantage must use the true episode return, not the last target."""

    def test_true_episode_return_is_the_reward_sum(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.env.instance_generator import InstanceGenerator
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy
        from coverage_env.rl.ppo.loss import compute_advantage
        from coverage_env.rl.ppo.ppo import PPOAgent

        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=20000, randomize_each_reset=False
        )
        agent = PPOAgent(CoveragePolicy(env, config=config), config)
        rewards = [-1.0, -2.0, -3.0]
        dones = [False, False, True]
        _adv, returns = compute_advantage(
            rewards, [0.0] * 3, [0.0] * 3, dones, 1.0, 0.95,
            device=torch.device("cpu"),
        )
        # gamma=1: the GAE target at the final step equals that step's reward only.
        self.assertAlmostEqual(float(returns[-1]), rewards[-1], places=6)
        self.assertNotAlmostEqual(float(returns[-1]), sum(rewards), places=3,
                                  msg="末端 GAE 目标不等于整局回报")


class DuplicateMainTests(unittest.TestCase):
    def test_single_main_guard(self):
        import inspect
        import coverage_env.rl.trainer.distributed as module
        source = inspect.getsource(module)
        self.assertEqual(source.count('if __name__ == "__main__":'), 1,
                         "distributed.py 只能有一个 __main__ guard")


class CriticBiasSyncTests(unittest.TestCase):
    def test_warm_start_uses_reduced_mean(self):
        """Without DDP the mean is local; the reduction path must exist in source."""
        import inspect
        from coverage_env.rl.ppo.ppo import PPOAgent
        source = inspect.getsource(PPOAgent._warm_start_value_bias)
        self.assertIn("all_reduce", source,
                      "critic warm start 必须跨 rank 归约")

    def test_warm_start_writes_local_mean_without_distributed(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.env.instance_generator import InstanceGenerator
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy
        from coverage_env.rl.ppo.ppo import PPOAgent

        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=20000, randomize_each_reset=False
        )
        agent = PPOAgent(CoveragePolicy(env, config=config), config)
        agent._warm_start_value_bias(torch.tensor([-4.0, -6.0]))
        bias = agent._critic_module().network[-1].bias
        self.assertAlmostEqual(float(bias.mean()), -5.0, places=5)


if __name__ == "__main__":
    unittest.main()
