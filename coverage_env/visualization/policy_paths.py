"""Draw the full plan the greedy policy executes on fixed test scenarios.

Answers "is the learned strategy actually sensible?" by drawing every leg of the
greedy plan on fixed instances with different field counts, side by side with the
nearest-unfinished-field heuristic on the *same* instances. Intended as a
post-training check after each milestone:

    PYTHONPATH=. python -m coverage_env.visualization.policy_paths \
        --checkpoint coverage_env/results/run_x/ckpt_update_100.pt \
        --field-counts 5 10 20 --out coverage_env/results/run_x/paths.png

The figure shows, per panel: field outlines, the depot, the UAV start, the flight
path coloured by visit order, entry points numbered in visit order, and resupply
returns as red dashed legs. It also prints the same numbers as a table so the
result can be read without opening the image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field as dataclass_field
from math import hypot
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as PolygonPatch  # noqa: E402

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from coverage_env.env.instance_generator import InstanceGenerator  # noqa: E402
from coverage_env.rl.decoder.action_decoder import CoveragePolicy  # noqa: E402
from coverage_env.rl.ppo.ppo import PPOAgent  # noqa: E402

DEFAULT_FIELD_COUNTS = (5, 10, 20)
BASE_SEED = 1000


@dataclass
class EpisodeTrace:
    """Everything needed to draw and score one greedy episode."""

    label: str
    num_fields: int
    seed: int
    reward: float = 0.0
    coverage: float = 0.0
    steps: int = 0
    resupplies: int = 0
    reason: str = ""
    start: tuple | None = None
    depot: tuple | None = None
    visits: list = dataclass_field(default_factory=list)
    resupply_points: list = dataclass_field(default_factory=list)


def greedy_chooser(actor):
    """Return a chooser that takes the policy's highest-probability action."""

    def choose(state, environment):
        with torch.no_grad():
            output = actor(state, environment=environment)
        distribution = output["action_distribution"]
        if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
            return None
        return output["actions"][int(distribution.argmax().item())]

    return choose


def nearest_field_chooser():
    """Return a chooser that flies to the nearest unfinished field."""

    def choose(state, environment):
        actions = environment.get_action_candidates()
        if not actions:
            return None
        mask = environment.get_action_mask(actions)
        position = environment.uavs[0].current_position
        best, best_distance = None, float("inf")
        for action, feasible in zip(actions, mask):
            if not feasible:
                continue
            field = environment.get_field(action.field_id)
            distance = hypot(
                field.centroid[0] - position[0], field.centroid[1] - position[1]
            )
            if distance < best_distance:
                best, best_distance = action, distance
        return best

    return choose


def run_episode(environment, choose, *, label, num_fields, seed, max_steps=600):
    """Execute one episode with ``choose`` and return its trace."""
    state = environment.reset()
    trace = EpisodeTrace(
        label=label,
        num_fields=num_fields,
        seed=seed,
        start=tuple(environment.uavs[0].current_position),
        depot=tuple(environment.depot.get_position()),
    )
    for _ in range(max_steps):
        action = choose(state, environment)
        if action is None:
            trace.reason = trace.reason or "no_action"
            break
        state, reward, done, info = environment.step(action)
        trace.reward += float(reward)
        trace.visits.append({
            "order": len(trace.visits) + 1,
            "field_id": action.field_id,
            "entry": action.entry_boundary_id,
            "exit": action.exit_boundary_id,
            "path": [tuple(point) for point in info.get("path", [])],
        })
        if info.get("auto_resupply"):
            trace.resupplies += 1
            trace.resupply_points.append(
                tuple(environment.depot.get_position())
            )
        if done:
            trace.reason = info.get("termination_reason", "done")
            break
    trace.steps = len(trace.visits)
    trace.coverage = float(environment.get_summary()["coverage_ratio"])
    return trace


def build_traces(checkpoint, field_counts, *, actor_seed=0):
    """Return ``{(num_fields, policy_label): {label: trace}}`` for one checkpoint."""
    config = load_config(DEFAULT_CONFIG_PATH)
    if checkpoint is not None:
        from coverage_env.rl.decoder.hierarchical_decoder import (
            checkpoint_decoder_flags,
        )

        saved = torch.load(checkpoint, map_location="cpu")
        config["rl"]["decoder"].update(checkpoint_decoder_flags(saved["actor"]))
    scenarios = {}
    for num_fields in field_counts:
        seed = BASE_SEED + num_fields
        environment = InstanceGenerator(config).generate(
            num_fields=num_fields, seed=seed, randomize_each_reset=False
        )

        torch.manual_seed(actor_seed)
        actor = CoveragePolicy(
            InstanceGenerator(config).generate(num_fields=num_fields, seed=seed),
            config=config,
        )
        if checkpoint:
            agent = PPOAgent(actor, config)
            agent.load_checkpoint(checkpoint, map_location="cpu")
        actor.eval()
        traces = {
            "greedy policy": run_episode(
                environment, greedy_chooser(actor),
                label="greedy policy", num_fields=num_fields, seed=seed,
            ),
            "nearest field": run_episode(
                environment, nearest_field_chooser(),
                label="nearest field", num_fields=num_fields, seed=seed,
            ),
        }
        scenarios[num_fields] = traces
    return scenarios


def _draw_panel(ax, environment, trace):
    """Draw one field layout plus the executed path, coloured by visit order."""
    for field in environment.fields:
        ax.add_patch(PolygonPatch(
            list(field.xy_polygon.exterior.coords), closed=True,
            facecolor="#eef3e8", edgecolor="#7c8b74", alpha=0.9, linewidth=0.8,
        ))
        ax.annotate(field.field_id, field.centroid, xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="#4a5843")

    depot = trace.depot or (0.0, 0.0)
    ax.scatter(*depot, marker="s", s=95, color="#8e44ad", zorder=5, label="depot")
    if trace.start is not None:
        ax.scatter(*trace.start, marker="*", s=190, color="#c0392b", zorder=6,
                   label="UAV start")

    total = max(len(trace.visits), 1)
    colour_map = plt.cm.viridis
    previous = trace.start
    for visit in trace.visits:
        path = visit["path"] or ([previous] if previous else [])
        colour = colour_map((visit["order"] - 1) / total)
        if len(path) > 1:
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            ax.plot(xs, ys, color=colour, linewidth=1.9, alpha=0.95, zorder=3)
        entry = path[-1] if path else previous
        if entry is not None:
            ax.annotate(str(visit["order"]), entry, xytext=(3, -9),
                        textcoords="offset points", fontsize=7.5,
                        color="#1a1a1a",
                        bbox=dict(boxstyle="circle,pad=0.14", fc="white",
                                  ec=colour, lw=1.0))
        previous = path[-1] if path else previous

    for point in trace.resupply_points:
        if previous is not None:
            ax.plot([previous[0], point[0]], [previous[1], point[1]],
                    color="#c0392b", linestyle=":", linewidth=1.6, zorder=4,
                    label="auto resupply")
        previous = point

    if previous is not None and depot is not None:
        ax.plot([previous[0], depot[0]], [previous[1], depot[1]],
                color="#7f8c8d", linestyle="--", linewidth=1.1, zorder=2,
                label="final return")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    ax.set_title(
        f"{trace.label} | {trace.num_fields} fields | reward {trace.reward:+.2f}\n"
        f"coverage {trace.coverage:.1%} | steps {trace.steps} | "
        f"resupplies {trace.resupplies} | {trace.reason}",
        fontsize=9,
    )
    ax.legend(loc="best", fontsize=6.5)


def plot_scenarios(scenarios, save_path=None):
    """Draw one row per field count, one column per policy."""
    field_counts = sorted(scenarios)
    policies = list(next(iter(scenarios.values())))
    figure, axes = plt.subplots(
        len(field_counts), len(policies),
        figsize=(6.0 * len(policies), 5.4 * len(field_counts)),
        squeeze=False,
    )
    config = load_config(DEFAULT_CONFIG_PATH)
    for row, num_fields in enumerate(field_counts):
        seed = BASE_SEED + num_fields
        environment = InstanceGenerator(config).generate(
            num_fields=num_fields, seed=seed, randomize_each_reset=False
        )
        for column, policy in enumerate(policies):
            _draw_panel(axes[row][column], environment, scenarios[num_fields][policy])
    figure.suptitle(
        "Greedy plans on fixed test scenarios (same instance per column pair)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return figure


def print_table(scenarios) -> None:
    """Print the same information as the figure, as a table."""
    print(f"{'fields':>7} {'policy':<15}{'reward':>10}{'coverage':>10}"
          f"{'steps':>7}{'resup':>7}  reason")
    for num_fields in sorted(scenarios):
        for policy, trace in scenarios[num_fields].items():
            print(f"{num_fields:>7} {policy:<15}{trace.reward:>10.2f}"
                  f"{trace.coverage:>10.1%}{trace.steps:>7}{trace.resupplies:>7}"
                  f"  {trace.reason}")


def main() -> None:
    """Parse arguments, run the scenarios, and write the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate; omit for an untrained policy.")
    parser.add_argument("--field-counts", type=int, nargs="+",
                        default=list(DEFAULT_FIELD_COUNTS))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    scenarios = build_traces(args.checkpoint, args.field_counts)
    print_table(scenarios)
    if args.out:
        plot_scenarios(scenarios, save_path=args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
