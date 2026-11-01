"""Plot one frozen test instance for every field count 5..20.

The overview figure is meant to make the scale range (field count and total
area) directly comparable; the representative figure gives 5/10/15/20 panels
more room.  All seeds come from the sealed final-test block 60000-61599.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.patches import Patch

from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.evaluation.maneuver_eval import (
    FIELD_COUNTS,
    TEST_BASE,
    protocol_config,
)

OUT = REPO / "coverage_env/results/maneuver_time/instance_figures"
FIG_COUNTS = [5, 10, 15, 20]


def sample_seed(count: int) -> int:
    """First seed of the frozen test block for this field count."""
    return TEST_BASE + (count - 5) * 100


def field_patch(field, color):
    exterior = list(field.xy_polygon.exterior.coords)
    patches = [PolygonPatch(exterior, closed=True, facecolor=color,
                            edgecolor="#274d31", linewidth=1.0, alpha=0.72,
                            zorder=2)]
    for ring in field.xy_polygon.interiors:
        patches.append(PolygonPatch(list(ring.coords), closed=True,
                                    facecolor="white", edgecolor="#274d31",
                                    linewidth=0.7, alpha=1.0, zorder=3))
    return patches


def draw_instance(ax, count: int, *, labels: bool = True):
    seed = sample_seed(count)
    environment = InstanceGenerator(protocol_config()).generate(
        num_fields=count, seed=seed,
        randomize_each_reset=False, randomize_uav=False,
    )
    fields = environment.fields
    palette = plt.get_cmap("tab20")
    for index, field in enumerate(fields):
        for patch in field_patch(field, palette(index % 20)):
            ax.add_patch(patch)
        if labels:
            cx, cy = field.centroid
            ax.text(cx, cy, str(index + 1), ha="center", va="center",
                    fontsize=5.2 if count >= 15 else 6.4,
                    color="#17351d", zorder=6,
                    bbox={"boxstyle": "circle,pad=0.10", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.62})
    depot = environment.depot.get_position()
    ax.scatter([depot[0]], [depot[1]], marker="s", s=42, c="#111111",
               edgecolor="white", linewidth=0.7, zorder=8)
    uav = environment.uavs[0]
    uav_pos = uav.current_position or depot
    ax.scatter([uav_pos[0]], [uav_pos[1]], marker="^", s=58, facecolor="none",
               edgecolor="#c62828", linewidth=1.4, zorder=9)
    total_area = sum(float(f.area) for f in fields)
    xs, ys = [], []
    for field in fields:
        minx, miny, maxx, maxy = field.bounding_box
        xs += [minx, maxx]; ys += [miny, maxy]
    xs.append(depot[0]); ys.append(depot[1])
    span_x = max(xs) - min(xs); span_y = max(ys) - min(ys)
    margin = 0.06 * max(span_x, span_y, 1.0)
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.grid(True, color="#d7ded8", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4, length=2.2, pad=1.5)
    ax.set_title(f"n={count}  |  {total_area/10000:.2f} ha  |  seed {seed}",
                 fontsize=8.6, pad=4, color="#1d3322")
    return environment, total_area


def overview():
    figure, axes = plt.subplots(4, 4, figsize=(17.5, 17.0))
    records = []
    for ax, count in zip(axes.flat, FIELD_COUNTS):
        environment, total_area = draw_instance(ax, count)
        records.append({
            "field_count": count,
            "seed": sample_seed(count),
            "total_area_m2": total_area,
            "depot_xy": list(environment.depot.get_position()),
            "uav_start_xy": list(environment.uavs[0].current_position or []),
            "field_areas_m2": [float(f.area) for f in environment.fields],
        })
    handles = [
        Patch(facecolor="#9ecf9a", edgecolor="#274d31", label="field"),
        Line2D([0], [0], marker="s", linestyle="", color="#111111", label="depot"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="none",
               markeredgecolor="#c62828", label="UAV start"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=10, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(
        "Frozen test instances across field counts (one per scale, seeds 60000-61599)",
        fontsize=15, y=0.995)
    figure.text(0.5, 0.973,
                "Per-field area 0.5–2.0 ha; shown ha is total polygon area; "
                "local metric coordinates; polygon numbers are field indices.",
                ha="center", va="top", fontsize=9, color="#4a5d4d")
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    path = OUT / "instances_5_to_20.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    figure.savefig(OUT / "instances_5_to_20.svg", bbox_inches="tight")
    figure.savefig(OUT / "instances_5_to_20.pdf", bbox_inches="tight")
    plt.close(figure)
    (OUT / "instances_5_to_20.json").write_text(
        json.dumps({"seed_block": [TEST_BASE, TEST_BASE + 1600 - 1],
                    "samples": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return path


def representative():
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 12.5))
    for ax, count in zip(axes.flat, FIG_COUNTS):
        draw_instance(ax, count)
    handles = [
        Patch(facecolor="#9ecf9a", edgecolor="#274d31", label="field"),
        Line2D([0], [0], marker="s", linestyle="", color="#111111", label="depot"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="none",
               markeredgecolor="#c62828", label="UAV start"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3,
                  frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle("Representative test instances: 5, 10, 15, 20 fields",
                    fontsize=15, y=0.995)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = OUT / "instances_5_10_15_20.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(OUT / "instances_5_10_15_20.pdf", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(overview())
    print(representative())
