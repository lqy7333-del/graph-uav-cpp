"""Tests for growth-edge region construction and boundary-aware sweeps."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from shapely.geometry import LineString, Point

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import GridPoint, discretize_field
from coverage_env.planner.boundary_aware_sweep import BoundaryAwareSweepPlanner
from coverage_env.planner.boundary_region_generator import BoundaryRegionGenerator
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.planner.growth_edge import GrowthEdgeSelector
from coverage_env.rl.action_space import CoverageAction
from coverage_env.rl.reward import calculate_reward


def make_field(field_id="field-1", vertices=None, resolution=5.0):
    transformer = CoordinateTransformer(31.23, 121.47)
    vertices = vertices or [(0, 0), (40, 0), (40, 30), (0, 30)]
    field = Field(field_id, [transformer.to_latlon(x, y) for x, y in vertices])
    discretize_field(field, resolution)
    return field


def make_environment(field):
    entry = field.get_boundary("B0")
    depot = Depot("depot-1", (
        entry.center[0] + entry.normal_vector[0] * 10,
        entry.center[1] + entry.normal_vector[1] * 10,
    ))
    uav = UAV(
        "UAV-1", 100_000, 8, 5, 2,
        energy_consumption_parameters=energy_parameters(),
    )
    environment = CoverageEnvironment([field], depot, [uav])
    environment.reset()
    return environment


class BoundaryGuidedPlannerTests(unittest.TestCase):
    def test_edge_seed_growth_order_has_connected_workload_prefixes(self):
        points = [
            GridPoint("root", 0.0, 0.0, "synthetic", 1.0, neighbors=["bridge"]),
            GridPoint("bridge", 1.0, 0.7, "synthetic", 1.0,
                      neighbors=["root", "tip"]),
            GridPoint("tip", 2.0, 0.1, "synthetic", 1.0, neighbors=["bridge"]),
        ]
        field = SimpleNamespace(grid_points=points, grid_resolution=1.0)
        edge = SimpleNamespace(
            id="edge", start_point=(-1.0, 0.0), end_point=(3.0, 0.0),
            center=(1.0, 0.0), normal_vector=(0.0, -1.0),
            tangent_vector=(1.0, 0.0),
        )
        order = BoundaryRegionGenerator()._expansion_order(field, edge)
        self.assertEqual({point.id for point in order}, {"root", "bridge", "tip"})
        for prefix_length in range(1, len(order) + 1):
            prefix = order[:prefix_length]
            ids = {point.id for point in prefix}
            reached = {prefix[0].id}
            pending = [prefix[0].id]
            by_id = {point.id: point for point in prefix}
            while pending:
                current = by_id[pending.pop()]
                for neighbor_id in current.neighbors:
                    if neighbor_id in ids and neighbor_id not in reached:
                        reached.add(neighbor_id)
                        pending.append(neighbor_id)
            self.assertEqual(reached, ids)

    def test_opposite_edges_grow_a_connected_region_from_intermediate_edge(self):
        field = make_field()
        entry, exit_edge = field.get_boundary("B0"), field.get_boundary("B2")
        selection = GrowthEdgeSelector().select(field, entry, exit_edge)
        self.assertNotIn(selection.growth_edge.id, {entry.id, exit_edge.id})
        self.assertTrue(selection.used_convex_geometry)

        generated = BoundaryRegionGenerator().generate(
            field, entry, exit_edge, 0.25
        )
        selected_ids = {point.id for point in generated.region}
        reached = {generated.region[0].id}
        pending = [generated.region[0].id]
        by_id = {point.id: point for point in generated.region}
        while pending:
            point = by_id[pending.pop()]
            for neighbor in point.neighbors:
                if neighbor in selected_ids and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        self.assertEqual(reached, selected_ids)
        self.assertGreaterEqual(generated.selected_area, generated.required_area)
        self.assertAlmostEqual(
            generated.max_reachable_area,
            sum(point.point_area for point in field.grid_points),
        )

    def test_adjacent_edges_still_choose_a_distinct_growth_edge(self):
        field = make_field()
        entry, exit_edge = field.get_boundary("B0"), field.get_boundary("B1")
        selection = GrowthEdgeSelector().select(field, entry, exit_edge)
        self.assertNotIn(selection.growth_edge.id, {entry.id, exit_edge.id})
        self.assertTrue(selection.candidate_ids)
        region = BoundaryRegionGenerator().generate(
            field, entry, exit_edge, 0.2
        ).region
        plan = BoundaryAwareSweepPlanner().plan(
            region, field, entry, exit_edge,
            current_position=(-10.0, -10.0), lane_spacing=5.0,
        )
        self.assertLess(
            LineString([entry.start_point, entry.end_point]).distance(
                Point(plan.entry_point)
            ),
            1e-7,
        )
        self.assertLess(
            LineString([exit_edge.start_point, exit_edge.end_point]).distance(
                Point(plan.exit_point)
            ),
            1e-7,
        )

    def test_random_convex_polygon_has_a_valid_growth_edge(self):
        field = make_field(
            vertices=[(0, 0), (45, 5), (55, 25), (32, 43), (8, 36), (-5, 18)]
        )
        entry, exit_edge = field.get_boundary("B1"), field.get_boundary("B4")
        selection = GrowthEdgeSelector().select(field, entry, exit_edge)
        self.assertTrue(selection.used_convex_geometry)
        self.assertNotIn(selection.growth_edge.id, {entry.id, exit_edge.id})
        generated = BoundaryRegionGenerator().generate(
            field, entry, exit_edge, 0.15
        )
        self.assertGreater(generated.selected_area, 0.0)
        plan = BoundaryAwareSweepPlanner().plan(
            generated.region, field, entry, exit_edge,
            current_position=(0.0, 0.0), lane_spacing=5.0,
        )
        self.assertTrue(plan.coverage_path)
        self.assertLess(
            LineString([entry.start_point, entry.end_point]).distance(
                Point(plan.entry_point)
            ),
            1e-7,
        )

    def test_sweep_jointly_respects_entry_and_exit_edges(self):
        field = make_field()
        entry, exit_edge = field.get_boundary("B0"), field.get_boundary("B3")
        region = BoundaryRegionGenerator().generate(
            field, entry, exit_edge, 0.35
        ).region
        planner = BoundaryAwareSweepPlanner()
        plan = planner.plan(
            region, field, entry, exit_edge,
            current_position=(-10.0, -10.0), lane_spacing=5.0,
        )
        entry_line = LineString([entry.start_point, entry.end_point])
        exit_line = LineString([exit_edge.start_point, exit_edge.end_point])
        self.assertLess(entry_line.distance(Point(plan.entry_point)), 1e-7)
        self.assertLess(exit_line.distance(Point(plan.exit_point)), 1e-7)
        self.assertGreater(len(plan.coverage_strokes), 1)
        self.assertEqual(plan.coverage_path[0], plan.coverage_strokes[0][0])
        self.assertEqual(plan.coverage_path[-1], plan.coverage_strokes[-1][1])
        self.assertLess(
            plan.total_cost,
            float("inf"),
        )

    def test_sweep_handles_a_clipped_region_narrower_than_swath(self):
        field = make_field()
        region = [GridPoint(
            id="thin-edge-cell", x=20.0, y=0.25, field_id=field.field_id,
            point_area=4.0,
        )]
        plan = BoundaryAwareSweepPlanner().plan(
            region, field, field.get_boundary("B0"), field.get_boundary("B2"),
            current_position=(20.0, -5.0), lane_spacing=5.0,
            candidate_directions=[0.0],
        )
        self.assertTrue(plan.coverage_path)
        self.assertEqual(len(plan.coverage_strokes), 1)

    def test_workload_beyond_growth_component_is_rejected_before_mutation(self):
        field = make_field()
        environment = make_environment(field)
        entry, exit_edge = field.get_boundary("B0"), field.get_boundary("B2")
        growth = GrowthEdgeSelector().select(field, entry, exit_edge).growth_edge
        edge_line = LineString([growth.start_point, growth.end_point])
        points = list(field.grid_points)
        near = min(points, key=lambda point: edge_line.distance(Point(point.x, point.y)))
        far = max(points, key=lambda point: edge_line.distance(Point(point.x, point.y)))
        remaining_ids = {near.id, far.id}
        for point in points:
            point.covered = point.id not in remaining_ids
        environment._action_estimate_cache.clear()

        action = CoverageAction(field.field_id, entry.id, exit_edge.id)
        maximum_ratio = environment.estimate_workload(action)
        self.assertGreater(maximum_ratio, 0.0)
        self.assertLess(maximum_ratio, 1.0)
        mask = environment.get_action_mask([action])
        self.assertEqual(mask, [1])  # Spatial action remains feasible for a partial strip.

        before = environment.get_state()
        with self.assertRaises(ValueError):
            environment.step(CoverageConfiguration(
                field.field_id, entry.id, exit_edge.id, 1.0
            ))
        self.assertEqual(environment.get_state(), before)

    def test_reward_penalizes_new_fragmentation_without_changing_action_shape(self):
        field = make_field()
        points = field.grid_points
        for point in points:
            point.covered = True
        points[0].covered = False
        points[-1].covered = False
        summary = CoverageEnvironment._fragmentation_summary(field, 1)
        self.assertEqual(summary["component_count"], 2)
        self.assertEqual(summary["small_component_count"], 2)

        weights = {"reward_weights": {
            "lambda1": 0.0, "lambda2": 0.0, "lambda3": 1.0, "lambda4": 0.5
        }}
        base = calculate_reward({"coverage": 10.0}, weights)
        penalized = calculate_reward({
            "coverage": 10.0, "fragmentation_penalty": 2.0
        }, weights)
        self.assertAlmostEqual(base - penalized, 1.0)
        self.assertFalse(hasattr(CoverageAction("f", "B0", "B1"), "workload_ratio"))


if __name__ == "__main__":
    unittest.main()
