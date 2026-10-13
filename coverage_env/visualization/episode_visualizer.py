"""Plot a full baseline episode and its configuration-level feedback."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box
from shapely.ops import unary_union


class EpisodeVisualizer:
    """Render fields, depot, coverage, configuration paths, and step labels."""

    def plot(
        self,
        environment,
        *,
        show: bool = False,
        save_path: str | Path | None = None,
        ax: Optional[Axes] = None,
    ) -> tuple[Figure, Axes]:
        """Create and optionally save a full episode visualization."""
        if ax is None:
            figure, ax = plt.subplots(figsize=(12, 9))
        else:
            figure = ax.figure

        for field in environment.fields:
            exterior = list(field.xy_polygon.exterior.coords)
            ax.fill(
                [point[0] for point in exterior], [point[1] for point in exterior],
                facecolor="#dcebd8", edgecolor="#436b43", alpha=0.55,
                linewidth=1.3,
                label="Fields" if field is environment.fields[0] else "_nolegend_",
                zorder=1,
            )
            for ring in field.xy_polygon.interiors:
                points = list(ring.coords)
                ax.fill(
                    [point[0] for point in points], [point[1] for point in points],
                    color="white", edgecolor="#436b43", linewidth=0.8, zorder=2,
                )

        depot_x, depot_y = environment.depot.position
        ax.scatter(
            [depot_x], [depot_y], marker="s", color="#222222", s=95,
            label="Depot", zorder=8,
        )
        colors = plt.get_cmap("tab20")
        field_step_counts: dict[object, int] = {}
        for step_index, record in enumerate(environment.execution_records, start=1):
            color = colors((step_index - 1) % 20)
            field = environment.get_field(record["field_id"])
            field_step_counts[field.field_id] = field_step_counts.get(field.field_id, 0) + 1
            region = record["region"]
            if region:
                cells = []
                for point in region:
                    half = sqrt(point.point_area) / 2.0
                    cells.append(box(
                        point.x - half, point.y - half,
                        point.x + half, point.y + half,
                    ))
                geometry = unary_union(cells).intersection(field.xy_polygon)
                parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
                for part in parts:
                    if part.is_empty or not hasattr(part, "exterior"):
                        continue
                    ax.add_patch(PolygonPatch(
                        list(part.exterior.coords), closed=True,
                        facecolor=color, edgecolor=color, linewidth=0.35,
                        alpha=0.34, zorder=3,
                    ))

            path = record["path"]
            if path:
                ax.plot(
                    [point[0] for point in path], [point[1] for point in path],
                    color=color, linewidth=1.1, marker=".", markersize=2.5,
                    label="_nolegend_", zorder=5,
                )
            entry = record["entry_point"]
            exit_point = record["exit_point"]
            growth_edge_id = record.get("growth_edge_id")
            for boundary_id, edge_color, edge_label in (
                (record.get("entry_boundary_id"), "#2ca02c", "Entry edge"),
                (record.get("exit_boundary_id"), "#1f77b4", "Exit edge"),
            ):
                if boundary_id is None:
                    continue
                boundary = field.get_boundary(boundary_id)
                ax.plot(
                    [boundary.start_point[0], boundary.end_point[0]],
                    [boundary.start_point[1], boundary.end_point[1]],
                    color=edge_color, linestyle="-.", linewidth=1.4,
                    alpha=0.7,
                    label=edge_label if step_index == 1 else "_nolegend_",
                    zorder=6,
                )
            if growth_edge_id is not None:
                growth_edge = field.get_boundary(growth_edge_id)
                ax.plot(
                    [growth_edge.start_point[0], growth_edge.end_point[0]],
                    [growth_edge.start_point[1], growth_edge.end_point[1]],
                    color="#b51fb5", linestyle=":", linewidth=1.5,
                    alpha=0.8, label="Growth edge" if step_index == 1 else "_nolegend_",
                    zorder=6,
                )
            ax.scatter(
                [entry[0]], [entry[1]], marker="^", color=color, s=36,
                edgecolor="white", linewidth=0.4, zorder=7,
            )
            ax.scatter(
                [exit_point[0]], [exit_point[1]], marker="s", color=color, s=28,
                edgecolor="white", linewidth=0.4, zorder=7,
            )
            local_step = field_step_counts[field.field_id]
            label_offsets = (
                (5, 5), (17, 5), (5, 16), (17, 16),
                (-17, 5), (-17, 16), (5, -15), (17, -15),
            )
            ax.annotate(
                str(step_index), entry,
                xytext=label_offsets[(local_step - 1) % len(label_offsets)],
                textcoords="offset points",
                fontsize=6, color="#222222", zorder=9,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.65},
            )

        # The per-configuration routes include the connector from the previous
        # UAV position. This overlay makes the complete mission trajectory clear.
        for uav_id, trajectory in environment.get_summary()["uav_trajectory"].items():
            if len(trajectory) > 1:
                ax.plot(
                    [point[0] for point in trajectory],
                    [point[1] for point in trajectory],
                    color="#333333", linewidth=0.65, linestyle="--",
                    alpha=0.55, label="_nolegend_", zorder=4,
                )

        # Add a readable close-up of one parcel so the overview remains useful
        # when many fields occupy a wide workspace.
        if environment.fields:
            zoom_field = environment.fields[0]
            zoom_ax = ax.inset_axes([0.66, 0.55, 0.32, 0.34])
            outline = list(zoom_field.xy_polygon.exterior.coords)
            zoom_ax.fill(
                [point[0] for point in outline], [point[1] for point in outline],
                facecolor="#dcebd8", edgecolor="#436b43", alpha=0.5, linewidth=1.0,
            )
            zoom_step = 0
            for step_index, record in enumerate(environment.execution_records, start=1):
                if record["field_id"] != zoom_field.field_id:
                    continue
                zoom_step += 1
                color = colors((step_index - 1) % 20)
                cells = []
                for point in record["region"]:
                    half = sqrt(point.point_area) / 2.0
                    cells.append(box(
                        point.x - half, point.y - half,
                        point.x + half, point.y + half,
                    ))
                if cells:
                    geometry = unary_union(cells).intersection(zoom_field.xy_polygon)
                    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
                    for part in parts:
                        if not part.is_empty and hasattr(part, "exterior"):
                            zoom_ax.add_patch(PolygonPatch(
                                list(part.exterior.coords), closed=True,
                                facecolor=color, edgecolor=color, alpha=0.35,
                                linewidth=0.3,
                            ))
                path = record["path"]
                if path:
                    zoom_ax.plot(
                        [point[0] for point in path], [point[1] for point in path],
                        color=color, linewidth=0.9, marker=".", markersize=2,
                    )
                remaining = record.get("remaining_points", [])
                if remaining:
                    zoom_ax.scatter(
                        [point.x for point in remaining],
                        [point.y for point in remaining],
                        color="#555555", marker=".", s=5, alpha=0.18,
                    )
                zoom_ax.annotate(
                    str(step_index), record["entry_point"], xytext=(3, 3),
                    textcoords="offset points", fontsize=6,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.7},
                )
            min_x, min_y, max_x, max_y = zoom_field.xy_polygon.bounds
            pad = max(max_x - min_x, max_y - min_y) * 0.12
            zoom_ax.set_xlim(min_x - pad, max_x + pad)
            zoom_ax.set_ylim(min_y - pad, max_y + pad)
            zoom_ax.set_aspect("equal", adjustable="box")
            zoom_ax.set_title(f"Close-up: {zoom_field.field_id} ({zoom_step} steps)", fontsize=8)
            zoom_ax.tick_params(labelsize=6)
            zoom_ax.grid(True, alpha=0.2)

        summary = environment.get_summary()
        ax.set_title(
            f"Coverage episode: {summary['execution_count']} steps, "
            f"{summary['coverage_ratio']:.1%} coverage, "
            f"{summary['total_distance']:.0f} m"
        )
        ax.set_xlabel("Shared local X (m)")
        ax.set_ylabel("Shared local Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        handles = [
            Patch(facecolor="#dcebd8", edgecolor="#436b43", label="Fields"),
            Patch(facecolor="#8c6bb1", edgecolor="#8c6bb1", alpha=0.34,
                  label="Coverage region"),
            Line2D([0], [0], color="#8c6bb1", linewidth=1.2,
                   label="Coverage path (step number at entry)"),
            Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="--",
                   label="UAV trajectory"),
            Line2D([0], [0], marker="^", linestyle="None", color="#555555",
                   label="Entry point"),
            Line2D([0], [0], marker="s", linestyle="None", color="#555555",
                   label="Exit point"),
            Line2D([0], [0], marker="s", linestyle="None", color="#222222",
                   label="Depot"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8, ncols=2)
        figure.tight_layout()
        if save_path is not None:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        return figure, ax
