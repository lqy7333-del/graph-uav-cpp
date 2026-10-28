"""structure_diff components must not double count and must close to the total."""
import unittest

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.evaluation.structure_diff import decompose


class StructureDiffAccountingTests(unittest.TestCase):
    def test_components_close_to_total(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=21013, randomize_each_reset=False
        )
        env.reset()
        # replay the nearest-neighbour sequence through decompose
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
        plan_env = InstanceGenerator(config).generate(
            num_fields=5, seed=21013, randomize_each_reset=False
        )
        plan_env.reset()
        sequence = []
        done = False
        while not done and len(sequence) < 400:
            action = _nearest_field_choice(plan_env, None)
            if action is None:
                break
            sequence.append((action.field_id, action.entry_boundary_id,
                             action.exit_boundary_id))
            _s, _r, done, _i = plan_env.step(action)
        out = decompose(env, sequence)
        self.assertIsNotNone(out)
        # Four disjoint simulator buckets, not center-distance proxies or subsets.
        parts = out["coverage"] + out["transit"] + out["resupply_steps"] + out["final_return"]
        self.assertAlmostEqual(parts, out["total"], delta=1e-3,
                               msg="四个互斥距离分项未闭合到 total")
        self.assertLessEqual(out["resupply_steps"] + out["final_return"], out["total"] + 1e-3)


if __name__ == "__main__":
    unittest.main()
