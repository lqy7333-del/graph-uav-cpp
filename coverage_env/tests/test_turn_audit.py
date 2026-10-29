"""Hand-computable turn geometry regressions."""
import unittest
from coverage_env.rl.evaluation.turn_audit import path_angles, stroke_reversals


class TurnAuditTests(unittest.TestCase):
    """Distinguish sweep reversals from connector corners and redundant points."""
    def test_redundant_points(self):
        self.assertEqual(path_angles([(0,0),(0,0),(1,0),(2,0)])['geometric_corners'],0)

    def test_one_sweep_reversal_two_corners(self):
        result=path_angles([(0,0),(10,0),(10,5),(0,5)])
        self.assertEqual(result['geometric_corners'],2)
        self.assertAlmostEqual(result['heading_change_deg'],180)
        self.assertEqual(stroke_reversals([((0,0),(10,0)),((10,5),(0,5))]),1)

    def test_collinear_fragments(self):
        self.assertEqual(stroke_reversals([((0,0),(1,0)),((2,0),(3,0))]),0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            path_angles([],0)
