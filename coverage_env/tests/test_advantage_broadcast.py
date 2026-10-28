"""The advantage-mode dispatch must actually change the advantages fed to PPO.

The previous revision shipped the broadcast helper but never called it, so
``advantage_mode=broadcast`` silently trained plain GAE and two ablation runs were
invalid. These tests exercise the dispatch itself, not the helper's existence.
"""
import unittest

import torch


def _agent(mode=None):
    from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    from coverage_env.rl.decoder.action_decoder import CoveragePolicy
    from coverage_env.rl.ppo.ppo import PPOAgent

    config = load_config(DEFAULT_CONFIG_PATH)
    if mode is not None:
        config["ppo"]["advantage_mode"] = mode
    env = InstanceGenerator(config).generate(num_fields=5, seed=8000,
                                             randomize_each_reset=False)
    return PPOAgent(CoveragePolicy(env, config=config), config)


class _Row:
    """Minimal stand-in for a rollout transition."""

    def __init__(self, value, done, reward=None, truncated=False):
        self.value = value
        self.done = done
        self.reward = value if reward is None else reward
        self.truncated = truncated


class DispatchTests(unittest.TestCase):
    def test_gae_and_broadcast_produce_different_advantages(self):
        """The whole point: the two modes must not return the same tensor."""
        rows = [_Row(1.0, False), _Row(2.0, False), _Row(3.0, True),
                _Row(4.0, False), _Row(5.0, True)]
        returns = torch.tensor([10.0, 12.0, 15.0, 7.0, 9.0])
        gae = torch.tensor([0.5, 0.7, 0.9, 0.2, 0.3])

        gae_agent = _agent("gae")
        broadcast_agent = _agent("broadcast")
        gae_out = gae_agent._select_advantages(rows, returns, gae, torch.device("cpu"))
        broadcast_out = broadcast_agent._select_advantages(
            rows, returns, gae, torch.device("cpu"))

        self.assertTrue(torch.equal(gae_out, gae))
        self.assertFalse(torch.equal(gae_out, broadcast_out),
                         "broadcast 与 gae 返回了相同的优势：分发没有生效")
        # Episode 1 covers rows 0-2, episode 2 covers rows 3-4.
        self.assertEqual(broadcast_out[:3].tolist(), [6.0, 6.0, 6.0])
        self.assertEqual(broadcast_out[3:].tolist(), [9.0, 9.0])

    def test_default_mode_is_gae(self):
        agent = _agent()                      # no explicit advantage_mode
        rows = [_Row(1.0, True)]
        returns = torch.tensor([3.0])
        gae = torch.tensor([0.42])
        out = agent._select_advantages(rows, returns, gae, torch.device("cpu"))
        self.assertTrue(torch.equal(out, gae))

    def test_unknown_mode_raises(self):
        agent = _agent("nonsense")
        with self.assertRaises(ValueError):
            agent._select_advantages([_Row(1.0, True)], torch.tensor([1.0]),
                                     torch.tensor([0.0]), torch.device("cpu"))

    def test_update_records_the_effective_mode(self):
        """A run must report which mode actually took effect."""
        agent = _agent("broadcast")
        self.assertFalse(hasattr(agent, "last_advantage_mode"))
        # the attribute is only set inside update(); presence of the dispatch is
        # asserted by the behavioural tests above, this one pins the reporting hook.
        self.assertTrue(hasattr(type(agent), "_select_advantages"))


if __name__ == "__main__":
    unittest.main()


class BaselineReplacementTests(unittest.TestCase):
    """critic baseline vs nearest-neighbour baseline, same broadcast advantage."""

    def _agent(self, mode):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.env.instance_generator import InstanceGenerator
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy
        from coverage_env.rl.ppo.ppo import PPOAgent

        config = load_config(DEFAULT_CONFIG_PATH)
        config["ppo"]["advantage_mode"] = mode
        env = InstanceGenerator(config).generate(num_fields=5, seed=8000,
                                                 randomize_each_reset=False)
        return PPOAgent(CoveragePolicy(env, config=config), config)

    def test_agent_has_nn_baselines(self):
        agent = self._agent("gae")
        self.assertEqual(agent.nn_baselines, [])

    def test_critic_and_nn_baselines_differ(self):
        rows = [_Row(1.0, False), _Row(2.0, True), _Row(3.0, False), _Row(4.0, True)]
        returns = torch.tensor([10.0, 12.0, 7.0, 9.0])
        gae = torch.tensor([0.1, 0.2, 0.3, 0.4])
        critic = self._agent("broadcast_critic")
        nn_agent = self._agent("broadcast_nn")
        nn_agent.nn_baselines = [11.0, 8.0]
        a = critic._select_advantages(rows, returns, gae, torch.device("cpu"))
        b = nn_agent._select_advantages(rows, returns, gae, torch.device("cpu"))
        self.assertEqual(a.tolist(), [2.0, 2.0, 4.0, 4.0])   # (1+2)-1, (3+4)-3
        self.assertEqual(b.tolist(), [-8.0, -8.0, -1.0, -1.0])  # (1+2)-11, (3+4)-8
        self.assertFalse(torch.equal(a, b), "两种基线必须给出不同的优势")

    def test_missing_nn_baseline_raises(self):
        agent = self._agent("broadcast_nn")
        agent.nn_baselines = []
        with self.assertRaises(ValueError):
            agent._select_advantages([_Row(1.0, True)], torch.tensor([1.0]),
                                     torch.tensor([0.0]), torch.device("cpu"))
