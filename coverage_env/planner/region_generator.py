"""Select a workload-sized prefix of an entry-reachable scan ordering."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, sqrt
from typing import Sequence

from coverage_env.geometry.discretization import GridPoint
from coverage_env.planner.coverage_path import order_grid_points

Point2D = tuple[float, float]


@dataclass(frozen=True)
class RegionSelection:
    """Coverage points and the full scan ordering for a reachable component."""

    region: list[GridPoint]
    scan_order: list[GridPoint]
    required_area: float
    selected_area: float


def generate_region(
    reachable_component: Sequence[GridPoint],
    orientation: float,
    workload_ratio: float,
    *,
    entry_point: Point2D | None = None,
    exit_point: Point2D | None = None,
    workload_reference_area: float | None = None,
) -> RegionSelection:
    """Select the leading scan-ordered cells within an entry-reachable component.

    The workload is a fraction of the reference remaining area (the reachable
    component area by default). If the entry component is smaller than the
    requested area, all reachable cells are selected and the action makes
    partial progress instead of crossing an uncovered-point gap.
    """
    if not isfinite(workload_ratio) or not 0.0 < workload_ratio <= 1.0:
        raise ValueError("workload_ratio must be finite and in (0, 1].")
    component = [point for point in reachable_component if not point.covered]
    if not component:
        raise ValueError("The entry boundary has no reachable uncovered grid points.")

    component_area = sum(point.point_area for point in component)
    reference_area = (
        component_area if workload_reference_area is None
        else float(workload_reference_area)
    )
    if not isfinite(reference_area) or reference_area <= 0.0:
        raise ValueError("workload_reference_area must be finite and positive.")
    required_area = min(workload_ratio * reference_area, component_area)
    theta = radians(orientation)
    normal = (-sin(theta), cos(theta))
    resolution = max(sqrt(point.point_area) for point in component)
    cross_track = {
        point.id: point.x * normal[0] + point.y * normal[1]
        for point in component
    }
    bands: dict[int, list[GridPoint]] = {}
    for point in component:
        band_id = round(cross_track[point.id] / resolution)
        bands.setdefault(band_id, []).append(point)

    start_cross = (
        entry_point[0] * normal[0] + entry_point[1] * normal[1]
        if entry_point is not None else min(cross_track.values())
    )
    target_cross = (
        exit_point[0] * normal[0] + exit_point[1] * normal[1]
        if exit_point is not None else start_cross
    )
    start_band = min(bands, key=lambda band: abs(
        sum(cross_track[p.id] for p in bands[band]) / len(bands[band]) - start_cross
    ))
    start_value = sum(cross_track[p.id] for p in bands[start_band]) / len(bands[start_band])
    direction = 1 if target_cross >= start_cross else -1
    ordered_bands = sorted(
        bands,
        key=lambda band: (
            0 if (sum(cross_track[p.id] for p in bands[band]) / len(bands[band]) - start_value) * direction >= -1e-9 else 1,
            abs(sum(cross_track[p.id] for p in bands[band]) / len(bands[band]) - start_value),
        ),
    )
    ordered_by_band = {
        band_id: order_grid_points(
            bands[band_id], orientation,
            start_point=entry_point, target_point=exit_point,
        )
        for band_id in ordered_bands
    }
    scan_order = [point for band_id in ordered_bands for point in ordered_by_band[band_id]]
    region: list[GridPoint] = []
    selected_area = 0.0
    for band_id in ordered_bands:
        for point in ordered_by_band[band_id]:
            region.append(point)
            selected_area += point.point_area
            if selected_area + 1e-9 >= required_area:
                break
        if selected_area + 1e-9 >= required_area:
            break
    return RegionSelection(region, scan_order, required_area, selected_area)
