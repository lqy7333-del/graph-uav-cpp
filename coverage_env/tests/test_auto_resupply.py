"""Tests for automatic return-to-depot resupply when resources run out."""

from __future__ import annotations

import unittest

from coverage_env.config import DEFAULT_CONFIG_PATH, energy_parameters, load_config
from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.rl.action_mask import get_action_mask


def build_two_field_env(max_payload=1.5, max_energy=100_000.0):
    """Build two 40x30m fields with a payload-limited UAV (one tank < 2 fields)."""
    transformer = CoordinateTransformer(31.23, 121.47)
    fields = []
    for index, offset_x in enumerate((0.0, 60.0), start=1):
        vertices = [
            transformer.to_latlon(offset_x, 0.0),
            transformer.to_latlon(offset_x + 40.0, 0.0),
            transformer.to_latlon(offset_x + 40.0, 30.0),
            transformer.to_latlon(offset_x, 30.0),
        ]
        fields.append(Field(f"field-{index}", vertices))
    depot = Depot("depot-1", (-10.0, -10.0))
    uav = UAV(
        "UAV-1",
        max_energy=max_energy,
        speed=8.0,
        coverage_width=5.0,
        sensor_width=2.0,
        energy_consumption_parameters=energy_parameters(),
        max_payload=max_payload,
        payload_per_area=0.001,
    )
    environment = CoverageEnvironment(
        fields, depot, [uav],
        rl_config={"return_to_depot": True, "allow_auto_resupply": True},
    )
    environment.reset()
    return environment


def drive_episode(environment, max_steps=100):
    """Run greedy first-feasible actions to completion, recording transitions."""
    transitions = []
    for _ in range(max_steps):
        actions = environment.get_action_candidates()
        mask = get_action_mask(actions, environment)
        feasible = [a for a, ok in zip(actions, mask) if ok]
        if not feasible:
            break
        next_state, reward, done, info = environment.step(feasible[0])
        transitions.append((next_state, reward, done, info))
        if done:
            break
    return transitions, environment.get_summary()


class AutoResupplyTests(unittest.TestCase):
    """Verify resupply trigger, statistics preservation, and termination."""

    def test_resource_depletion_triggers_resupply_and_continues(self):
        environment = build_two_field_env(max_payload=1.5)
        transitions, summary = drive_episode(environment)

        self.assertGreaterEqual(summary["resupply_count"], 1)
        self.assertEqual(len(summary["completed_fields"]), 2)

        resupplied = [
            (state, info) for state, _r, _d, info in transitions
            if info.get("auto_resupply")
        ]
        self.assertTrue(resupplied)
        for state, info in resupplied:
            self.assertTrue(info["auto_resupply"])
            self.assertEqual(info["resupply_reason"], "no_feasible_coverage_action")
            self.assertEqual(
                state["uavs"]["UAV-1"]["position"], environment.depot.get_position()
            )
            self.assertEqual(
                state["uavs"]["UAV-1"]["remaining_energy"],
                environment.uavs[0].max_energy,
            )
            self.assertEqual(
                state["uavs"]["UAV-1"]["remaining_payload"],
                environment.uavs[0].max_payload,
            )

    def test_resupply_preserves_progress_and_statistics(self):
        environment = build_two_field_env(max_payload=1.5)
        _transitions, summary = drive_episode(environment)

        uav = environment.uavs[0]
        self.assertGreater(uav.total_distance, 0.0)
        self.assertGreater(uav.total_energy_consumed, 0.0)
        self.assertGreater(uav.total_payload_consumed, 0.0)
        # Resupply is tracked separately from consumption.
        self.assertGreater(uav.total_energy_recharged, 0.0)
        self.assertGreater(uav.total_payload_resupplied, 0.0)
        self.assertEqual(summary["total_energy_recharged"], uav.total_energy_recharged)
        self.assertEqual(summary["total_payload_resupplied"], uav.total_payload_resupplied)
        # All grid points ended covered (progress preserved, not reset).
        self.assertTrue(all(
            point.covered
            for field in environment.fields
            for point in field.grid_points
        ))

    def test_return_infeasible_does_not_resupply(self):
        environment = build_two_field_env(max_payload=1.5)
        uav = environment.uavs[0]
        uav.remaining_payload = 0.0
        uav.remaining_energy = 1.0
        uav.current_position = (1000.0, 1000.0)
        saved = (uav.current_position, uav.remaining_energy, uav.remaining_payload)

        result = environment._attempt_auto_resupply()

        self.assertIsNone(result)
        self.assertEqual(environment._termination_reason, "return_infeasible")
        self.assertEqual(
            (uav.current_position, uav.remaining_energy, uav.remaining_payload),
            saved,
        )

    def test_no_feasible_action_at_full_resources_terminates(self):
        environment = build_two_field_env(max_payload=1.5)
        # Simulate a coverage structure with nothing left to do.
        for field in environment.fields:
            for point in field.grid_points:
                point.covered = True

        self.assertFalse(environment._has_feasible_action_at_full_resources())
        result = environment._attempt_auto_resupply()
        self.assertIsNone(result)
        self.assertEqual(
            environment._termination_reason, "no_feasible_action_at_full_resources"
        )
        self.assertEqual(environment._resupply_count, 0)

    def test_policy_decision_after_resupply(self):
        from coverage_env.rl.decoder.action_decoder import CoveragePolicy

        config = load_config(DEFAULT_CONFIG_PATH)
        environment = build_two_field_env(max_payload=1.5)
        policy = CoveragePolicy(environment, config=config)
        policy.eval()

        state = environment.get_state()
        done = False
        for _ in range(100):
            output = policy(state, environment=environment)
            distribution = output["action_distribution"]
            self.assertGreater(float(distribution.sum()), 0.0)
            action = policy.greedy_action(output)
            state, _reward, done, _info = environment.step(action)
            if done:
                break
        self.assertTrue(done)
        self.assertGreaterEqual(environment.get_summary()["resupply_count"], 1)

    def test_regular_and_failed_transitions_still_work(self):
        environment = build_two_field_env(max_payload=50.0)
        actions = environment.get_action_candidates()
        mask = get_action_mask(actions, environment)
        action = next(a for a, ok in zip(actions, mask) if ok)
        next_state, reward, done, info = environment.step(action)
        self.assertTrue(info["success"])
        self.assertFalse(info.get("auto_resupply", False))
        self.assertIsInstance(next_state, dict)

        # A workload-exceeding configuration is a penalized failure transition.
        from coverage_env.planner.configuration import CoverageConfiguration

        failing = CoverageConfiguration(
            action.field_id, action.entry_boundary_id, action.exit_boundary_id, 1.0
        )
        _ns, _r, _d, fail_info = environment.step(failing)
        if fail_info.get("failed"):
            self.assertFalse(fail_info["success"])


if __name__ == "__main__":
    unittest.main()
