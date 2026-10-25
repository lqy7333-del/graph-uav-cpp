import unittest, torch
class BroadcastTests(unittest.TestCase):
    def test_episode_returns_broadcast_to_its_steps(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.rl.ppo.ppo import PPOAgent
        from coverage_env.env.instance_generator import InstanceGenerator
        c = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(c).generate(num_fields=5, seed=8000, randomize_each_reset=False)
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy
        agent = PPOAgent(CoveragePolicy(env, config=c), c)
        class Row:
            def __init__(self, done): self.done = done
        transitions = [Row(False), Row(False), Row(True), Row(False), Row(True)]
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out = agent._broadcast_episode_returns(transitions, returns, torch.device("cpu"))
        self.assertEqual(out.tolist(), [3.0, 3.0, 3.0, 5.0, 5.0])

    def test_default_mode_is_gae(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        c = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(c["ppo"].get("advantage_mode", "gae"), "gae")
if __name__ == "__main__":
    unittest.main()
