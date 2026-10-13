"""Tests for real payload consumption and transfer/coverage path separation."""

from __future__ import annotations

import unittest

from coverage_env.config import DEFAULT_CONFIG_PATH, energy_parameters, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.env.uav import UAV
from coverage_env.rl.action_mask import get_action_mask


def make_uav(*, max_energy=1000.0, max_payload=100.0, payload_per_area=0.001) -> UAV:
    """Create a UAV with explicit payload parameters for direct execution tests."""
    return UAV(
        "UAV-1",
        max_energy=max_energy,
        speed=8.0,
        coverage_width=5.0,
        sensor_width=2.0,
        energy_consumption_parameters=energy_parameters(),
        max_payload=max_payload,
        payload_per_area=payload_per_area,
    )


class PayloadAndTransferSeparationTests(unittest.TestCase):
    """Verify transfer charges only energy while coverage charges energy+payload."""

    def test_transfer_consumes_no_payload(self):
        uav = make_uav()
        uav.reset((0.0, 0.0))
        result = uav.execute_path([(0, 0), (100, 0)], area=0.0)
        self.assertAlmostEqual(result["payload_consumption"], 0.0)
        self.assertAlmostEqual(result["energy_consumption"], uav.alpha * 100.0)
        self.assertEqual(uav.remaining_payload, uav.max_payload)

    def test_coverage_consumes_payload_and_energy(self):
        uav = make_uav()
        uav.reset((0.0, 0.0))
        result = uav.execute_path([(0, 0), (10, 0)], area=1000.0)
        self.assertAlmostEqual(result["payload_consumption"], 0.001 * 1000.0)
        self.assertAlmostEqual(
            result["energy_consumption"], uav.alpha * 10.0 + uav.beta * 1000.0
        )
        self.assertAlmostEqual(result["remaining_payload"], 100.0 - 1.0)

    def test_executor_separates_transfer_and_coverage(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        environment = InstanceGenerator(config).generate(num_fields=5, seed=7)
        environment.reset()
        actions = environment.get_action_candidates()
        mask = get_action_mask(actions, environment)
        action = next(action for action, ok in zip(actions, mask) if ok)
        _next_state, _reward, _done, info = environment.step(action)

        record = environment.execution_records[-1]
        gamma = environment.uavs[0].payload_per_area
        self.assertAlmostEqual(record["payload"], gamma * info["coverage_gain"])
        self.assertAlmostEqual(
            record["transfer_distance"] + record["coverage_distance"],
            record["distance"],
        )
        self.assertAlmostEqual(
            environment.uavs[0].remaining_payload,
            environment.uavs[0].max_payload - record["payload"],
        )

    def test_insufficient_payload_raises_and_masks(self):
        uav = make_uav()
        uav.reset((0.0, 0.0))
        uav.remaining_payload = 0.5
        with self.assertRaises(ValueError):
            uav.execute_path([(0, 0), (10, 0)], area=1000.0)

        config = load_config(DEFAULT_CONFIG_PATH)
        environment = InstanceGenerator(config).generate(num_fields=5, seed=7)
        environment.reset()
        actions = environment.get_action_candidates()
        mask = get_action_mask(actions, environment)
        action = next(action for action, ok in zip(actions, mask) if ok)
        self.assertGreater(environment.estimate_workload(action), 0.0)
        environment.uavs[0].remaining_payload = 0.0
        self.assertEqual(environment.estimate_workload(action), 0.0)


if __name__ == "__main__":
    unittest.main()
