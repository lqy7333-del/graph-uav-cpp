"""Matplotlib visualization for field geometry and derived boundaries."""

from __future__ import annotations

from pathlib import Path
from math import sqrt
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as PolygonPatch
from shapely.geometry import box
from shapely.ops import unary_union

from coverage_env.env.field import Field


def plot_field(
    field: Field,
    *,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Draw a field, boundary ids, centroid, and major orientation axis.

    Returns the figure and axes so callers can customize or save the plot.
    Set ``show=False`` for tests, notebooks, or headless batch jobs.
    """
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
    else:
        figure = ax.figure

    exterior = list(field.xy_polygon.exterior.coords)
    ax.fill(
        [point[0] for point in exterior],
        [point[1] for point in exterior],
        color="#acd8a0",
        alpha=0.68,
        edgecolor="#315b36",
        linewidth=2.0,
        label="Field",
    )
    for ring in field.xy_polygon.interiors:
        points = list(ring.coords)
        ax.fill(
            [point[0] for point in points],
            [point[1] for point in points],
            color="white",
            edgecolor="#315b36",
            linewidth=1.4,
        )

    diagonal = max(
        ((field.bounding_box[2] - field.bounding_box[0]) ** 2
         + (field.bounding_box[3] - field.bounding_box[1]) ** 2) ** 0.5,
        1.0,
    )
    label_offset = diagonal * 0.018
    for boundary in field.boundaries:
        ax.plot(
            [boundary.start_point[0], boundary.end_point[0]],
            [boundary.start_point[1], boundary.end_point[1]],
            color="#244a8f" if not boundary.is_hole else "#8b5a2b",
            linewidth=2.0,
        )
        label_point = (
            boundary.center[0] + boundary.normal_vector[0] * label_offset,
            boundary.center[1] + boundary.normal_vector[1] * label_offset,
        )
        ax.text(
            label_point[0],
            label_point[1],
            boundary.id,
            color="#152b53",
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.8},
        )

    ax.scatter(
        [field.centroid[0]], [field.centroid[1]],
        color="#c43b32", marker="*", s=150, zorder=4, label="Centroid"
    )
    arrow_length = diagonal * 0.20
    ax.annotate(
        "",
        xy=(field.centroid[0] + field.orientation_vector[0] * arrow_length,
            field.centroid[1] + field.orientation_vector[1] * arrow_length),
        xytext=field.centroid,
        arrowprops={"arrowstyle": "-|>", "color": "#e07b16", "lw": 2.5},
        zorder=5,
    )
    ax.plot([], [], color="#e07b16", linewidth=2.5,
            label=f"Orientation: {field.orientation:.1f}°")
    ax.set_title(f"Field {field.field_id} geometry")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


def plot_grid(
    field: Field,
    *,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Draw a field outline together with its discretized grid points."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
    else:
        figure = ax.figure

    exterior = list(field.xy_polygon.exterior.coords)
    ax.fill(
        [point[0] for point in exterior],
        [point[1] for point in exterior],
        color="#acd8a0",
        alpha=0.35,
        edgecolor="#315b36",
        linewidth=2.0,
        label="Field boundary",
        zorder=1,
    )
    for ring in field.xy_polygon.interiors:
        points = list(ring.coords)
        ax.fill(
            [point[0] for point in points],
            [point[1] for point in points],
            color="white",
            edgecolor="#315b36",
            linewidth=1.4,
            zorder=2,
        )

    uncovered = [point for point in field.grid_points if not point.covered]
    covered = [point for point in field.grid_points if point.covered]
    if uncovered:
        ax.scatter(
            [point.x for point in uncovered],
            [point.y for point in uncovered],
            s=12,
            color="#2355a6",
            label="Uncovered grid point",
            zorder=3,
        )
    if covered:
        ax.scatter(
            [point.x for point in covered],
            [point.y for point in covered],
            s=14,
            color="#e24a33",
            label="Covered grid point",
            zorder=4,
        )

    ax.set_title(
        f"Field {field.field_id} grid"
        + (f" (resolution={field.grid_resolution:g} m)"
           if field.grid_resolution is not None else " (not discretized)")
    )
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if field.grid_points:
        ax.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


def plot_uav(
    depot,
    uav,
    *,
    fields: Optional[list[Field]] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Draw field outlines, depot, UAV position, and its latest executed path."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
    else:
        figure = ax.figure

    for field in fields or []:
        exterior = list(field.xy_polygon.exterior.coords)
        ax.fill(
            [point[0] for point in exterior],
            [point[1] for point in exterior],
            color="#acd8a0",
            alpha=0.25,
            edgecolor="#315b36",
            linewidth=1.5,
            label=f"Field {field.field_id}",
            zorder=1,
        )

    depot_position = depot.get_position()
    ax.scatter(
        [depot_position[0]], [depot_position[1]],
        marker="s", s=90, color="#333333", label="Depot", zorder=5,
    )
    if uav.current_path:
        ax.plot(
            [point[0] for point in uav.current_path],
            [point[1] for point in uav.current_path],
            color="#e07b16",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Executed path",
            zorder=3,
        )
    if uav.current_position is not None:
        ax.scatter(
            [uav.current_position[0]], [uav.current_position[1]],
            marker="^", s=110, color="#c43b32", label=f"UAV {uav.uav_id}", zorder=6,
        )

    ax.set_title(f"UAV {uav.uav_id} simulation")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


def plot_configuration_execution(
    field: Field,
    entry_point,
    exit_point,
    region,
    coverage_path,
    *,
    entry_transit_path=None,
    exit_transit_path=None,
    coverage_start_point=None,
    coverage_strokes=None,
    scan_transit_paths=None,
    growth_edge_id: str | None = None,
    remaining_points=None,
    entry_boundary_id: str | None = None,
    exit_boundary_id: str | None = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Draw a configuration's selected region, entry/exit, and executed path."""
    if ax is None:
        figure, ax = plt.subplots(figsize=(9, 7))
    else:
        figure = ax.figure

    exterior = list(field.xy_polygon.exterior.coords)
    ax.fill(
        [point[0] for point in exterior],
        [point[1] for point in exterior],
        color="#acd8a0",
        alpha=0.35,
        edgecolor="#315b36",
        linewidth=2.0,
        label="Field boundary",
        zorder=1,
    )
    for ring in field.xy_polygon.interiors:
        points = list(ring.coords)
        ax.fill(
            [point[0] for point in points],
            [point[1] for point in points],
            color="white",
            edgecolor="#315b36",
            linewidth=1.3,
            zorder=2,
        )

    for boundary_id, color, label in (
        (entry_boundary_id, "#2ca02c", "Entry edge"),
        (exit_boundary_id, "#1f77b4", "Exit edge"),
    ):
        if boundary_id is None:
            continue
        boundary = field.get_boundary(boundary_id)
        ax.plot(
            [boundary.start_point[0], boundary.end_point[0]],
            [boundary.start_point[1], boundary.end_point[1]],
            color=color, linestyle="-.", linewidth=2.7,
            label=f"{label} {boundary_id}", zorder=6,
        )

    if growth_edge_id is not None:
        growth_edge = field.get_boundary(growth_edge_id)
        ax.plot(
            [growth_edge.start_point[0], growth_edge.end_point[0]],
            [growth_edge.start_point[1], growth_edge.end_point[1]],
            color="#b51fb5", linestyle="--", linewidth=3.0,
            label=f"Growth edge {growth_edge_id}", zorder=6,
        )
    if remaining_points:
        ax.scatter(
            [point.x for point in remaining_points],
            [point.y for point in remaining_points],
            color="#555555", marker=".", s=10, alpha=0.5,
            label="Remaining uncovered cells", zorder=4,
        )

    if region:
        cell_polygons = []
        for point in region:
            half_side = sqrt(point.point_area) / 2.0
            cell_polygons.append(
                box(point.x - half_side, point.y - half_side,
                    point.x + half_side, point.y + half_side)
            )
        covered_geometry = unary_union(cell_polygons).intersection(field.xy_polygon)
        geometries = (
            list(covered_geometry.geoms)
            if hasattr(covered_geometry, "geoms")
            else [covered_geometry]
        )
        first_region_patch = True
        for geometry in geometries:
            if geometry.is_empty or not hasattr(geometry, "exterior"):
                continue
            ax.add_patch(
                PolygonPatch(
                    list(geometry.exterior.coords),
                    closed=True,
                    facecolor="#f4cf62",
                    edgecolor="#b58c1f",
                    linewidth=0.6,
                    alpha=0.72,
                    label="Coverage region" if first_region_patch else None,
                    zorder=3,
                )
            )
            first_region_patch = False

    if entry_transit_path is not None or exit_transit_path is not None:
        if coverage_strokes:
            for stroke_index, (start, end) in enumerate(coverage_strokes):
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color="black", linewidth=2.0,
                    label="Coverage sweep" if stroke_index == 0 else None,
                    zorder=5,
                )
            for connector_index, connector in enumerate(scan_transit_paths or []):
                ax.plot(
                    [point[0] for point in connector],
                    [point[1] for point in connector],
                    color="#e58b24", linestyle="--", linewidth=1.8,
                    label="In-field transfer" if connector_index == 0 else None,
                    zorder=5,
                )
        else:
            ax.plot(
                [point[0] for point in coverage_path or []],
                [point[1] for point in coverage_path or []],
                color="black", linewidth=2.0, marker="o", markersize=3,
                label="Coverage path", zorder=5,
            )
        for route, color, label in (
            (entry_transit_path or [], "red", "Entry transit"),
            (exit_transit_path or [], "purple", "Exit transit"),
        ):
            if route:
                ax.plot(
                    [point[0] for point in route],
                    [point[1] for point in route],
                    color=color, linewidth=2.0, marker="o", markersize=3,
                    label=label, zorder=5,
                )
    elif coverage_path:
        ax.plot(
            [point[0] for point in coverage_path],
            [point[1] for point in coverage_path],
            color="#d62728",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="Coverage path",
            zorder=4,
        )
        route_start = coverage_path[0]
        if route_start != entry_point:
            ax.scatter(
                [route_start[0]], [route_start[1]],
                color="#7b3294", marker="^", s=95, edgecolor="white",
                linewidth=0.8, label="UAV start / depot", zorder=7,
            )
    ax.scatter(
        [entry_point[0]], [entry_point[1]],
        color="#2ca02c", marker="o", s=85, edgecolor="white",
        linewidth=0.8, label="Boundary entry", zorder=7,
    )
    if coverage_start_point is not None:
        ax.scatter(
            [coverage_start_point[0]], [coverage_start_point[1]],
            color="#f2c300", marker="o", s=85, edgecolor="black",
            linewidth=0.7, label="Coverage anchor", zorder=8,
        )
    ax.scatter(
        [exit_point[0]], [exit_point[1]],
        color="#1f77b4", marker="o", s=85, edgecolor="white",
        linewidth=0.8, label="Boundary exit", zorder=7,
    )
    ax.set_title(f"Field {field.field_id} configuration execution")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


def plot_mission(
    environment,
    *,
    ax: Optional[Axes] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Draw all fields and configuration paths from a completed mission.

    Each configuration receives a distinct Matplotlib cycle color. Coverage
    cells, path, entry, and exit share that color so consecutive actions are
    easy to distinguish in a multi-field task.
    """
    if ax is None:
        figure, ax = plt.subplots(figsize=(11, 8))
    else:
        figure = ax.figure

    for field in environment.fields:
        exterior = list(field.xy_polygon.exterior.coords)
        ax.fill(
            [point[0] for point in exterior], [point[1] for point in exterior],
            color="#acd8a0", alpha=0.25, edgecolor="#315b36", linewidth=1.8,
            label=f"Field {field.field_id}", zorder=1,
        )
        for ring in field.xy_polygon.interiors:
            points = list(ring.coords)
            ax.fill(
                [point[0] for point in points], [point[1] for point in points],
                color="white", edgecolor="#315b36", linewidth=1.0, zorder=2,
            )

    depot_position = environment.depot.get_position()
    ax.scatter(
        [depot_position[0]], [depot_position[1]], marker="s", s=100,
        color="#222222", edgecolor="white", linewidth=0.8,
        label="Depot", zorder=8,
    )

    color_map = plt.get_cmap("tab10")
    for index, record in enumerate(environment.execution_records):
        color = color_map(index % 10)
        field = environment.get_field(record["field_id"])
        region = record["region"]
        if region:
            cells = []
            for point in region:
                half_side = sqrt(point.point_area) / 2.0
                cells.append(box(
                    point.x - half_side, point.y - half_side,
                    point.x + half_side, point.y + half_side,
                ))
            geometry = unary_union(cells).intersection(field.xy_polygon)
            parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
            first_patch = True
            for part in parts:
                if part.is_empty or not hasattr(part, "exterior"):
                    continue
                ax.add_patch(PolygonPatch(
                    list(part.exterior.coords), closed=True, facecolor=color,
                    edgecolor=color, linewidth=0.4, alpha=0.38,
                    label=f"Coverage {index + 1}" if first_patch else None,
                    zorder=3,
                ))
                first_patch = False

        path = record["path"]
        if path:
            ax.plot(
                [point[0] for point in path], [point[1] for point in path],
                color=color, linewidth=1.7, marker="o", markersize=2.5,
                label=f"Configuration {index + 1}: {record['field_id']}",
                zorder=5,
            )
        ax.scatter(
            [record["entry_point"][0]], [record["entry_point"][1]],
            color=color, marker="^", s=55, edgecolor="white", linewidth=0.6,
            zorder=7,
        )
        ax.scatter(
            [record["exit_point"][0]], [record["exit_point"][1]],
            color=color, marker="s", s=45, edgecolor="white", linewidth=0.6,
            zorder=7,
        )

    summary = environment.get_summary()
    ax.set_title(
        "Mission execution: "
        f"{summary['execution_count']} configurations, "
        f"{summary['total_distance']:.1f} m, "
        f"{summary['coverage_ratio']:.1%} field coverage"
    )
    ax.set_xlabel("Shared local X (m)")
    ax.set_ylabel("Shared local Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax


def plot_graph(environment, graph, *, show: bool = True, save_path: str | None = None):
    """Visualize Field/Boundary nodes and the three permitted edge families."""
    figure, ax = plt.subplots(figsize=(12, 9))
    node_positions = {}
    field_count = len(environment.fields)
    for field_index, field in enumerate(environment.fields):
        patch = PolygonPatch(
            list(field.xy_polygon.exterior.coords), closed=True,
            facecolor="#e8efdf", edgecolor="#697b5d", alpha=0.65,
        )
        ax.add_patch(patch)
        node_positions[field_index] = field.centroid
        ax.scatter(*field.centroid, marker="o", s=105, color="#236b3b", zorder=4)
        ax.annotate(str(field.field_id), field.centroid, xytext=(5, 6),
                    textcoords="offset points", fontsize=9, color="#174d2a")

    for boundary_index, (field, boundary) in enumerate(
        (pair for field in environment.fields for pair in
         ((field, boundary) for boundary in field.boundaries))
    ):
        node_index = field_count + boundary_index
        node_positions[node_index] = boundary.center
        ax.scatter(*boundary.center, marker="^", s=44, color="#315caa", zorder=4)
        ax.annotate(f"{field.field_id}:{boundary.id}", boundary.center,
                    xytext=(3, 3), textcoords="offset points", fontsize=6,
                    color="#23457d")

    styles = {
        0: ("#617849", "--", "Field–Boundary"),
        1: ("#dc8a22", ":", "Boundary intra"),
        2: ("#8e44ad", "-", "Boundary inter"),
    }
    edge_index = graph.edge_index.detach().cpu().tolist()
    edge_types = graph.edge_type.detach().cpu().tolist()
    labeled_types = set()
    for (source, target), edge_type in zip(zip(*edge_index), edge_types):
        color, linestyle, label = styles[int(edge_type)]
        ax.plot(
            [node_positions[source][0], node_positions[target][0]],
            [node_positions[source][1], node_positions[target][1]],
            color=color, linestyle=linestyle, linewidth=0.8, alpha=0.42,
            label=label if edge_type not in labeled_types else None,
            zorder=2,
        )
        labeled_types.add(int(edge_type))

    ax.scatter([], [], marker="o", color="#236b3b", label="Field node")
    ax.scatter([], [], marker="^", color="#315caa", label="Boundary node")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.set_title(
        "Field–Boundary Heterogeneous Graph\n"
        f"nodes={graph.num_nodes}, edges={graph.edge_index.shape[1]}, "
        f"relation counts={graph.relation_counts}"
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=8)
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, ax
