"""A/B training curves, validation curves, and per-scale NN deltas (v2)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coverage_env.rl.evaluation.maneuver_eval import (
    _make_environment,
    _nn_choice,
    protocol_config,
    run_episode,
    validation_seeds,
)

ROOT = REPO / "coverage_env/results/maneuver_time"
V2 = ROOT / "formal_fixed_gate_v2"
OLD = ROOT / "formal_fixed"
GATE = ROOT / "formal_fixed_gate"
OUT = ROOT / "ab_curves"
SEEDS = [42, 43, 44]
COUNTS = list(range(5, 21))
UPDATES = list(range(20, 401, 20))
COLORS = {"A": "#2ca02c", "B": "#d62728"}
PRETTY = {"A": "A (distance)", "B": "B (time)"}
SMOOTH = 20


def sources_for(run: str) -> list[Path]:
    if run == "A_seed42":
        return [OLD / "A_seed42/train_log.csv",
                GATE / "A_seed42/train_log.csv",
                V2 / "A_seed42/train_log.csv"]
    return [V2 / run / "train_log.csv"]


def load_train(run: str) -> list[dict]:
    by_update: dict[int, dict] = {}
    for path in sources_for(run):
        for row in csv.DictReader(path.open()):
            by_update[int(row["update"])] = row
    rows = [by_update[u] for u in sorted(by_update)]
    assert [int(r["update"]) for r in rows] == list(range(1, 401)), run
    return rows


def trailing_mean(values, window: int = SMOOTH):
    """Backward mean over the last <=window updates; no edge padding."""
    array = np.asarray(values, dtype=float)
    out = np.empty_like(array)
    for i in range(len(array)):
        out[i] = float(array[max(0, i - window + 1):i + 1].mean())
    return out


def validation_curve(run: str):
    rows = json.loads((V2 / "validation" / run / "validation_rows.json").read_text())
    by_update = {}
    for update in UPDATES:
        subset = [r for r in rows if r["checkpoint_update"] == update]
        completed = [r for r in subset if r["completed"]]
        scale_means = []
        for count in COUNTS:
            values = [r["total_time"] for r in completed if r["field_count"] == count]
            if values:
                scale_means.append(float(np.mean(values)))
        by_update[update] = {
            "completed": len(completed),
            "total": len(subset),
            "scale_equal_time": float(np.mean(scale_means)) if scale_means else None,
        }
    selection = json.loads((V2 / "validation" / run / "selection.json").read_text())
    return by_update, int(selection["selected_update"])


def nn_validation():
    cache = OUT / "validation_nn_rows.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for count in COUNTS:
        for seed in validation_seeds()[count]:
            environment = _make_environment(protocol_config(), count, seed)
            metrics = run_episode(environment, _nn_choice)
            rows.append({**metrics, "field_count": count, "seed": seed})
    cache.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = {f"{objective}_seed{seed}": load_train(f"{objective}_seed{seed}")
             for seed in SEEDS for objective in ("A", "B")}
    validation = {}
    selected = {}
    for seed in SEEDS:
        for objective in ("A", "B"):
            run = f"{objective}_seed{seed}"
            validation[run], selected[run] = validation_curve(run)

    nn_rows = nn_validation()
    nn_by_instance = {(r["field_count"], r["seed"]): r["total_time"]
                      for r in nn_rows if r["completed"]}
    nn_scale_means = {
        count: float(np.mean([nn_by_instance[(count, seed)]
                              for seed in validation_seeds()[count]]))
        for count in COUNTS
    }
    nn_overall = float(np.mean(list(nn_scale_means.values())))

    # Per-scale NN - policy deltas; per-seed values retained for faint lines.
    deltas = {}
    for seed in SEEDS:
        for objective in ("A", "B"):
            run = f"{objective}_seed{seed}"
            rows = json.loads((V2 / "validation" / run / "validation_rows.json").read_text())
            by_instance = {(r["field_count"], r["seed"], r["checkpoint_update"]): r
                           for r in rows if r["completed"]}
            deltas[run] = {}
            for update in UPDATES:
                deltas[run][update] = {}
                for count in COUNTS:
                    values = []
                    for instance_seed in validation_seeds()[count]:
                        policy = by_instance.get((count, instance_seed, update))
                        nn = nn_by_instance.get((count, instance_seed))
                        if policy is not None and nn is not None:
                            values.append(nn - policy["total_time"])
                    deltas[run][update][count] = float(np.mean(values)) if values else None

    # ---------------- training curves: A and B in separate rows ----------------
    for metric, path_name, log_y in (
        ("mean_ep_reward", "ab_reward_curves", False),
        ("critic_loss", "ab_critic_curves", True),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=True)
        for row, objective in enumerate(("A", "B")):
            row_values = []
            for col, seed in enumerate(SEEDS):
                run = f"{objective}_seed{seed}"
                updates = np.array([int(r["update"]) for r in train[run]])
                values = np.array([float(r[metric]) for r in train[run]])
                row_values.append(values)
                color = COLORS[objective]
                axes[row, col].plot(updates, values, color=color, alpha=0.16, lw=0.7)
                axes[row, col].plot(updates, trailing_mean(values), color=color,
                                    lw=1.8)
                axes[row, col].set_title(f"{PRETTY[objective]} / seed {seed}",
                                          fontsize=10)
                axes[row, col].grid(True, alpha=0.25, which="both")
                axes[row, col].set_xlim(1, 400)
                if log_y:
                    axes[row, col].set_yscale("log")
                if col == 0:
                    axes[row, col].set_ylabel(
                        "mean completed-episode reward" if not log_y else "critic loss")
                if row == 1:
                    axes[row, col].set_xlabel("update")
            low = min(float(v.min()) for v in row_values)
            high = max(float(v.max()) for v in row_values)
            pad = 0.04 * max(high - low, 1.0)
            for col in range(3):
                if not log_y:
                    axes[row, col].set_ylim(low - pad, high + pad)
                else:
                    axes[row, col].set_ylim(max(low * 0.75, 1e-3), high * 1.2)
        fig.suptitle(
            "A/B training curves (raw faint; trailing 20-update mean, no edge padding)",
            fontsize=15)
        if metric == "mean_ep_reward":
            subtitle = (
                "rank0 completed-episode raw reward; A and B are shown separately. "
                "Rank0/complete-only metric, not a global task mean.")
        else:
            subtitle = (
                "rank0 critic loss, log scale; A and B are shown separately. "
                "The y scale is objective-specific and not a stability test.")
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=8.7, color="#4a5d4d")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"{path_name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"{path_name}.svg", bbox_inches="tight")
        plt.close(fig)

    # ---------------- validation total time, with NN level ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    for ax, seed in zip(axes, SEEDS):
        for objective in ("A", "B"):
            run = f"{objective}_seed{seed}"
            data = validation[run]
            updates = [u for u in UPDATES if data[u]["scale_equal_time"] is not None]
            values = [data[u]["scale_equal_time"] for u in updates]
            color = COLORS[objective]
            ax.plot(updates, values, color=color, marker="o", ms=3.4,
                    lw=1.3, label=PRETTY[objective])
            chosen = selected[run]
            ax.scatter([chosen], [data[chosen]["scale_equal_time"]], marker="*",
                       s=180, c=color, edgecolor="white", linewidth=0.8, zorder=6)
            ax.annotate(f"{chosen}", (chosen, data[chosen]["scale_equal_time"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7.2, color=color)
        ax.axhline(nn_overall, color="#444444", ls="--", lw=1.2, label="NN")
        ax.set_title(f"seed {seed}", fontsize=11)
        ax.set_xlabel("checkpoint update")
        ax.set_ylabel("scale-equal validation total time (s)" if seed == 42 else "")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 410)
    axes[0].legend(loc="best", fontsize=8.4)
    fig.suptitle("Validation time curves (96 fixed instances, all complete; lower is better)",
                 fontsize=14)
    fig.text(0.5, 0.02,
             f"NN level = {nn_overall:.1f} s (same 96 instances). Stars mark selected checkpoints; "
             "time metric includes maneuver loss and excludes unmodeled service/landing.",
             ha="center", fontsize=8.2, color="#4a5d4d")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUT / "ab_validation_time_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ab_validation_time_curves.svg", bbox_inches="tight")
    plt.close(fig)

    # ---------------- per-scale NN - policy curves ----------------
    fig, axes = plt.subplots(4, 4, figsize=(16.0, 14.0), sharex=True)
    for index, count in enumerate(COUNTS):
        ax = axes.flat[index]
        for objective in ("A", "B"):
            color = COLORS[objective]
            individual = []
            for seed in SEEDS:
                run = f"{objective}_seed{seed}"
                vals = [deltas[run][u][count] for u in UPDATES]
                individual.append(vals)
                ax.plot(UPDATES, vals, color=color, alpha=0.17, lw=0.8)
            mean = np.mean(np.asarray(individual, dtype=float), axis=0)
            ax.plot(UPDATES, mean, color=color, lw=1.8, label=PRETTY[objective])
        ax.axhline(0.0, color="#444444", lw=1.0, ls=":")
        ax.set_title(f"{count} fields", fontsize=9.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(1, 400)
        if index % 4 == 0:
            ax.set_ylabel("NN − policy (s)\npositive = policy faster", fontsize=7.2)
        if index >= 12:
            ax.set_xlabel("checkpoint update", fontsize=7.4)
        ax.tick_params(labelsize=6.5)
    handles = [
        plt.Line2D([0], [0], color=COLORS["A"], lw=2, label="A (distance), mean of seeds"),
        plt.Line2D([0], [0], color=COLORS["B"], lw=2, label="B (time), mean of seeds"),
        plt.Line2D([0], [0], color="#777777", lw=0.8, alpha=0.5, label="individual seed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Per-scale validation delta: NN − policy (positive = policy faster)",
                 fontsize=14, y=0.995)
    fig.text(0.5, 0.972,
             "Each panel uses the same 6 validation instances per scale; faded lines are "
             "individual training seeds, bold lines are seed means.",
             ha="center", fontsize=8.6, color="#4a5d4d")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "ab_validation_per_scale_nn_delta.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ab_validation_per_scale_nn_delta.svg", bbox_inches="tight")
    plt.close(fig)

    # ---------------- audit data ----------------
    training_stats = {}
    for run, rows in train.items():
        rewards = [float(r["mean_ep_reward"]) for r in rows]
        critics = [float(r["critic_loss"]) for r in rows]
        evs = [float(r["explained_variance"]) for r in rows]
        training_stats[run] = {
            "first50_reward_mean": float(np.mean(rewards[:50])),
            "last50_reward_mean": float(np.mean(rewards[-50:])),
            "last50_reward_std": float(np.std(rewards[-50:], ddof=1)),
            "last_raw_reward": float(rewards[-1]),
            "last_trailing20_reward": float(trailing_mean(rewards)[-1]),
            "first50_critic_mean": float(np.mean(critics[:50])),
            "last50_critic_mean": float(np.mean(critics[-50:])),
            "last50_ev_mean": float(np.mean(evs[-50:])),
        }
    validation_points = {}
    for run, data in validation.items():
        validation_points[run] = {
            str(u): {"scale_equal_time": data[u]["scale_equal_time"],
                     "completed": data[u]["completed"], "total": data[u]["total"]}
            for u in UPDATES
        }
    per_scale_mean = {}
    for run, data in deltas.items():
        per_scale_mean[run] = {
            str(u): {str(c): data[u][c] for c in COUNTS} for u in UPDATES
        }
    payload = {
        "seeds": SEEDS,
        "smooth_window": SMOOTH,
        "training_stats": training_stats,
        "validation_points": validation_points,
        "selected_update": selected,
        "nn_validation": {
            "scale_equal_mean": nn_overall,
            "per_scale_mean": {str(c): nn_scale_means[c] for c in COUNTS},
            "completed": sum(bool(r["completed"]) for r in nn_rows),
            "total": len(nn_rows),
        },
        "per_scale_delta": per_scale_mean,
        "data_note": ("mean_ep_reward/critic_loss are rank0, completed-episode logs; "
                      "validation uses 96 fixed instances and scale-equal means."),
    }
    (OUT / "ab_curves_data.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    with (OUT / "ab_training_logs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "update", "mean_ep_reward", "critic_loss",
                         "explained_variance", "entropy"])
        for run, rows in train.items():
            for row in rows:
                writer.writerow([run, row["update"], row["mean_ep_reward"],
                                 row["critic_loss"], row["explained_variance"],
                                 row["entropy"]])
    print(OUT / "ab_reward_curves.png")
    print(OUT / "ab_critic_curves.png")
    print(OUT / "ab_validation_time_curves.png")
    print(OUT / "ab_validation_per_scale_nn_delta.png")


if __name__ == "__main__":
    main()
