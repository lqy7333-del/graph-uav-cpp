"""Route maps for NN and the selected distance/time policies on 3 fixed scales.

Policies are evaluated with the same greedy deployment rule used by the final
test.  A and B use the same training seed (42) so the objective comparison is
paired; seed 42 is the first registered seed, not selected by final performance.
"""
from __future__ import annotations

import copy
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

from coverage_env.rl.evaluation.maneuver_eval import (
    FIELD_COUNTS,
    TEST_BASE,
    _make_environment,
    _nn_choice,
    _policy_choice_factory,
    load_policy,
    protocol_config,
    run_episode,
)

OUT = REPO / "coverage_env/results/maneuver_time/policy_routes"
SCALES = [5, 10, 20]
ROOT = REPO / "coverage_env/results/maneuver_time/formal_fixed_gate_v2"
SELECTIONS = {
    "A_seed42": ROOT / "validation/A_seed42/selection.json",
    "B_seed42": ROOT / "validation/B_seed42/selection.json",
}
COLORS = {"NN": "#1f77b4", "A_seed42": "#2ca02c", "B_seed42": "#d62728"}
PRETTY = {"NN": "NN", "A_seed42": "A (distance)", "B_seed42": "B (time)"}
FIELD_COLOR = "#e4eee2"


def sample_seed(count: int) -> int:
    return TEST_BASE + (count - 5) * 100


def field_patches(ax, environment):
    for index, field in enumerate(environment.fields):
        exterior = list(field.xy_polygon.exterior.coords)
        ax.add_patch(PolygonPatch(
            exterior, closed=True, facecolor=FIELD_COLOR,
            edgecolor="#6d8a70", linewidth=0.9, zorder=1))
        for ring in field.xy_polygon.interiors:
            ax.add_patch(PolygonPatch(
                list(ring.coords), closed=True, facecolor="white",
                edgecolor="#6d8a70", linewidth=0.6, zorder=1))
        cx, cy = field.centroid
        ax.text(cx, cy, str(index + 1), ha="center", va="center",
                fontsize=5.4 if len(environment.fields) >= 15 else 6.2,
                color="#526455", zorder=12,
                bbox={"boxstyle": "circle,pad=0.10", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.65})
    depot = environment.depot.get_position()
    ax.scatter([depot[0]], [depot[1]], marker="s", s=44, c="#111111",
               edgecolor="white", linewidth=0.7, zorder=8)


def row_bounds(environment):
    xs, ys = [], []
    for field in environment.fields:
        minx, miny, maxx, maxy = field.bounding_box
        xs += [minx, maxx]
        ys += [miny, maxy]
    depot = environment.depot.get_position()
    xs.append(depot[0]); ys.append(depot[1])
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = 0.07 * span
    return (min(xs) - margin, max(xs) + margin,
            min(ys) - margin, max(ys) + margin)


def draw_route(ax, route, color, *, label=None, start=True, end=True):
    if not route:
        return
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    ax.plot(xs, ys, color=color, linewidth=1.45, alpha=0.92,
            solid_capstyle="round", zorder=5, label=label)
    # Direction arrows at roughly 1/4, 1/2 and 3/4 of the path.
    for frac in (0.25, 0.5, 0.75):
        idx = min(len(route) - 2, max(0, int(frac * (len(route) - 1))))
        if idx < 0 or idx + 1 >= len(route):
            continue
        ax.annotate("", xy=route[idx + 1], xytext=route[idx],
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": 1.2, "mutation_scale": 9},
                    zorder=7)
    if start:
        ax.scatter([xs[0]], [ys[0]], marker="o", s=34, facecolor="white",
                   edgecolor=color, linewidth=1.5, zorder=9)
    if end:
        ax.scatter([xs[-1]], [ys[-1]], marker="*", s=95, c=color,
                   edgecolor="white", linewidth=0.6, zorder=10)


def run_policy(count, seed, policy_name):
    environment = _make_environment(protocol_config(), count, seed)
    if policy_name == "NN":
        choose = _nn_choice
    else:
        selection = json.loads(SELECTIONS[policy_name].read_text())["selected_checkpoint"]
        policy, _config, _device = load_policy(selection)
        choose = _policy_choice_factory(policy)
    metrics = run_episode(environment, choose)
    route = environment.get_summary()["uav_trajectory"]["UAV-1"]
    return environment, metrics, route


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    policies = ["NN", "A_seed42", "B_seed42"]
    rows = []
    records = {}
    for count in SCALES:
        seed = sample_seed(count)
        template = _make_environment(protocol_config(), count, seed)
        bounds = row_bounds(template)
        row = []
        for policy_name in policies:
            environment, metrics, route = run_policy(count, seed, policy_name)
            row.append((policy_name, environment, metrics, route))
            records.setdefault(str(count), {})[policy_name] = {
                "seed": seed,
                "completed": metrics["completed"],
                "total_time_s": metrics["total_time"],
                "distance_m": metrics["total_distance"],
                "visits": metrics["visits"],
                "coverage": metrics["coverage_ratio"],
                "points": route,
            }
        rows.append((count, seed, bounds, row, template))

    figure, axes = plt.subplots(3, 3, figsize=(15.0, 14.2))
    for row_index, (count, seed, bounds, row, template) in enumerate(rows):
        for col, (policy_name, environment, metrics, route) in enumerate(row):
            ax = axes[row_index, col]
            field_patches(ax, environment)
            draw_route(ax, route, COLORS[policy_name])
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            ax.grid(True, color="#dbe3dc", linewidth=0.45, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5, length=2.2, pad=1.5)
            status = "complete" if metrics["completed"] else "INCOMPLETE"
            ax.set_title(
                f"{PRETTY[policy_name]}  |  {metrics['total_time']:.1f} s  |  "
                f"{metrics['total_distance']/1000:.2f} km\n"
                f"{status}  |  visits {metrics['visits']}  |  coverage "
                f"{100*metrics['coverage_ratio']:.1f}%",
                fontsize=8.4, pad=4, color="#1d3322")
            if col == 0:
                ax.set_ylabel(f"n={count}, seed {seed}\nLocal Y (m)", fontsize=8.2)
            else:
                ax.set_ylabel("Local Y (m)", fontsize=7.2)
            ax.set_xlabel("Local X (m)", fontsize=7.2)
        axes[row_index, 0].text(
            0.02, 0.98, f"{count} fields", transform=axes[row_index, 0].transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            color="#1d3322", bbox={"boxstyle": "round,pad=0.25",
                                   "facecolor": "white", "edgecolor": "#b9c9bb"})

    handles = [
        Patch(facecolor=FIELD_COLOR, edgecolor="#6d8a70", label="field"),
        Line2D([0], [0], marker="s", linestyle="", color="#111111", label="depot"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#555555", label="route start"),
        Line2D([0], [0], marker="*", linestyle="", color="#555555", label="route end"),
        Line2D([0], [0], color=COLORS["NN"], lw=2, label="NN"),
        Line2D([0], [0], color=COLORS["A_seed42"], lw=2, label="A (distance)"),
        Line2D([0], [0], color=COLORS["B_seed42"], lw=2, label="B (time)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
                  fontsize=8.6, bbox_to_anchor=(0.5, 0.006))
    figure.suptitle(
        "NN vs trained policies on three fixed test instances (greedy deployment)",
        fontsize=15, y=0.995)
    figure.text(0.5, 0.974,
                "Same seed (42) for A/B, chosen before viewing these routes; "
                "depot square, field numbers, route arrows show travel direction.",
                ha="center", va="top", fontsize=8.8, color="#4a5d4d")
    figure.tight_layout(rect=(0, 0.03, 1, 0.965))
    path = OUT / "routes_3scales_3policies.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(OUT / "routes_3scales_3policies.svg", bbox_inches="tight")
    plt.close(figure)

    # Overlay panel for each scale: same routes, direct comparison.
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 7.0))
    for ax, (count, seed, bounds, row, template) in zip(axes, rows):
        field_patches(ax, row[0][1])
        for policy_name, environment, metrics, route in row:
            draw_route(ax, route, COLORS[policy_name], label=PRETTY[policy_name])
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.grid(True, color="#dbe3dc", linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5, length=2.2, pad=1.5)
        ax.set_title(f"n={count} | seed {seed}", fontsize=10)
        ax.set_xlabel("Local X (m)", fontsize=7.5)
        ax.set_ylabel("Local Y (m)", fontsize=7.5)
        ax.legend(loc="best", fontsize=7.0, frameon=True, framealpha=0.85)
    figure.suptitle("Route overlay: NN vs A (distance) vs B (time)",
                    fontsize=14, y=1.0)
    figure.tight_layout()
    overlay = OUT / "routes_overlay_3scales.png"
    figure.savefig(overlay, dpi=190, bbox_inches="tight")
    plt.close(figure)

    (OUT / "routes_metrics.json").write_text(
        json.dumps({"test_seed_block": [TEST_BASE, TEST_BASE + 1600 - 1],
                    "scales": SCALES, "policies": policies,
                    "records": records}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(path)
    print(overlay)


if __name__ == "__main__":
    main()
