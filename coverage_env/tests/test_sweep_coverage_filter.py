from types import SimpleNamespace
from unittest.mock import patch
from shapely.geometry import box
from coverage_env.planner.boundary_aware_sweep import BoundaryAwareSweepPlanner


def test_incomplete_cheapest_axis_cannot_hide_complete_axis():
    planner = BoundaryAwareSweepPlanner()
    region = [SimpleNamespace(x=0., y=0., point_area=4., coordinate=(0., 0.))]
    field = SimpleNamespace(xy_polygon=box(-10, -10, 10, 10))
    def lanes(geometry, angle, spacing, points):
        y = 4. if angle == 0 else 0.
        return [(y, [((-1., y), (1., y))])]
    def materialize(strokes, angle, *args):
        return SimpleNamespace(total_cost=1. if angle == 0 else 10., orientation=angle)
    with patch.object(planner, '_scan_intervals', side_effect=lanes), \
         patch.object(planner, '_choose_stroke_order', side_effect=lambda strokes, *a: (strokes, 0.)), \
         patch.object(planner, '_materialize_strokes', side_effect=materialize):
        kwargs = dict(current_position=(0., 0.), lane_spacing=2., candidate_directions=[0, 90])
        assert planner.plan(region, field, None, None, **kwargs).orientation == 0
        assert planner.plan(region, field, None, None, require_point_coverage=True, **kwargs).orientation == 90
