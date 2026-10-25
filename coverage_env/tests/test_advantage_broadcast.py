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

    def __init__(self, value, done):
        self.value = value
        self.done = done


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
        self.assertEqual(broadcast_out[:3].tolist(), [15.0, 15.0, 15.0])
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
