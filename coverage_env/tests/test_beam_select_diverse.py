"""Tests for field-diverse frontier retention (cost-to-go trace mode only)."""
import unittest


class SelectBeamTests(unittest.TestCase):
    def _node(self, value, field, gate=0):
        return (value, [(field, f"B{gate}", "B0")], object())

    def test_default_keeps_pure_reward_order(self):
        from coverage_env.rl.evaluation.beam_oracle import select_beam
        frontier = [self._node(-1.0, "field-A", 0), self._node(-1.1, "field-A", 1),
                    self._node(-9.0, "field-B", 0)]
        chosen = select_beam(frontier, 2, diverse_frontier=False)
        self.assertEqual([n[1][0][0] for n in chosen], ["field-A", "field-A"])

    def test_diverse_keeps_one_leader_per_field(self):
        """Five feasible fields must not be squeezed out by one field's gates."""
        from coverage_env.rl.evaluation.beam_oracle import select_beam
        frontier = [self._node(-1.0 - 0.01 * k, "field-A", k) for k in range(5)]
        frontier += [self._node(-5.0, "field-B", 0), self._node(-6.0, "field-C", 0),
                     self._node(-7.0, "field-D", 0), self._node(-8.0, "field-E", 0)]
        chosen = select_beam(frontier, 5, diverse_frontier=True)
        fields = {n[1][0][0] for n in chosen}
        self.assertEqual(len(fields), 5, f"首层 beam 被单一地块占满: {fields}")
        self.assertEqual(fields, {"field-A", "field-B", "field-C", "field-D", "field-E"})

    def test_diverse_fills_leftover_by_reward(self):
        from coverage_env.rl.evaluation.beam_oracle import select_beam
        # Two fields, three slots: the leaders take two, the best leftover takes the rest.
        frontier = [self._node(-1.0, "field-A", 0), self._node(-1.2, "field-A", 1),
                    self._node(-2.0, "field-B", 0)]
        chosen = select_beam(frontier, 3, diverse_frontier=True)
        self.assertEqual([round(n[0], 1) for n in chosen], [-1.0, -2.0, -1.2])

    def test_beam_search_default_still_available(self):
        import inspect
        from coverage_env.rl.evaluation.beam_oracle import beam_search
        signature = inspect.signature(beam_search)
        self.assertFalse(signature.parameters["diverse_frontier"].default)


if __name__ == "__main__":
    unittest.main()
