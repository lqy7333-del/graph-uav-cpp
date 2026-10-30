"""Unit tests for the frozen evaluation/selection protocol helpers."""

import copy
import unittest

import torch

from coverage_env.rl.checkpoint import parameter_sha256, state_dict_sha256
from coverage_env.rl.evaluation import maneuver_eval as protocol


class SeedBlockTests(unittest.TestCase):
    def test_validation_block_is_six_per_scale_and_contiguous(self):
        seeds = protocol.validation_seeds()
        self.assertEqual(tuple(seeds), protocol.FIELD_COUNTS)
        flat = [seed for count in protocol.FIELD_COUNTS for seed in seeds[count]]
        self.assertEqual(flat, list(range(40000, 40096)))
        self.assertTrue(all(len(seeds[count]) == 6 for count in protocol.FIELD_COUNTS))
        self.assertEqual(seeds[20][-1], 40095)

    def test_test_block_is_one_hundred_per_scale_and_contiguous(self):
        seeds = protocol.test_seeds()
        flat = [seed for count in protocol.FIELD_COUNTS for seed in seeds[count]]
        self.assertEqual(flat, list(range(50000, 51600)))
        self.assertTrue(all(len(seeds[count]) == 100 for count in protocol.FIELD_COUNTS))
        self.assertEqual(seeds[20][-1], 51599)

    def test_checkpoint_updates_are_twenty_to_four_hundred(self):
        self.assertEqual(protocol.expected_checkpoint_updates(),
                         list(range(20, 401, 20)))


class ProtocolConfigTests(unittest.TestCase):
    def test_frozen_values_pass(self):
        config = protocol.protocol_config()
        protocol._check_protocol_config(config)

    def test_changed_speed_is_rejected(self):
        config = protocol.protocol_config()
        config["uav"]["speed"] = 9.0
        with self.assertRaises(ValueError):
            protocol._check_protocol_config(config)

    def test_attention_must_be_disabled(self):
        config = protocol.protocol_config()
        config["rl"]["decoder"]["candidate_attention"] = True
        with self.assertRaises(ValueError):
            protocol._check_protocol_config(config)


class SelectionTests(unittest.TestCase):
    def _candidate(self, update, completed, mean=None):
        return {
            "update": update,
            "completed": completed,
            "total": 96,
            "completion_rate": completed / 96,
            "all_complete": completed == 96 and mean is not None,
            "mean_total_time": mean,
            "scale_mean_total_time": {} if mean is None else {5: mean},
            "per_scale_completed": {},
        }

    def test_complete_candidates_rank_by_scale_equal_mean_then_earlier(self):
        candidates = [
            self._candidate(20, 96, 100.0),
            self._candidate(40, 96, 90.0),
            self._candidate(60, 96, 90.0),
        ]
        selection = protocol.select_candidate(candidates, 96)
        self.assertTrue(selection["reliability_gate_passed"])
        self.assertEqual(selection["selected_update"], 40)

    def test_incomplete_candidates_never_compare_mean_time(self):
        candidates = [
            self._candidate(20, 90, 10.0),
            self._candidate(40, 95, 999.0),
            self._candidate(60, 95, 1.0),
        ]
        selection = protocol.select_candidate(candidates, 96)
        self.assertFalse(selection["reliability_gate_passed"])
        self.assertEqual(selection["selected_update"], 40)
        self.assertEqual(selection["rule"],
                         "completion_count_then_earlier_update_no_time_comparison")

    def test_completion_count_beats_any_time_value(self):
        candidates = [
            self._candidate(20, 96, 1000.0),
            self._candidate(40, 95, 1.0),
        ]
        selection = protocol.select_candidate(candidates, 96)
        self.assertTrue(selection["reliability_gate_passed"])
        self.assertEqual(selection["selected_update"], 20)


class HashTests(unittest.TestCase):
    def test_parameter_hash_changes_after_update(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        before = parameter_sha256(model)
        with torch.no_grad():
            model.weight.add_(0.01)
        self.assertNotEqual(before, parameter_sha256(model))

    def test_state_dict_hash_is_order_independent(self):
        tensors = {"b": torch.arange(4.0), "a": torch.ones(2)}
        reordered = {"a": tensors["a"].clone(), "b": tensors["b"].clone()}
        self.assertEqual(state_dict_sha256(tensors), state_dict_sha256(reordered))

    def test_state_dict_hash_detects_value_change(self):
        tensors = {"a": torch.ones(2)}
        changed = copy.deepcopy(tensors)
        changed["a"][0] = 2.0
        self.assertNotEqual(state_dict_sha256(tensors), state_dict_sha256(changed))


if __name__ == "__main__":
    unittest.main()
