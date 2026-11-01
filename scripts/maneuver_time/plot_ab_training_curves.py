"""A/B training and validation curves for the corrected v2 runs."""
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

ROOT = REPO / "coverage_env/results/maneuver_time"
V2 = ROOT / "formal_fixed_gate_v2"
OLD = ROOT / "formal_fixed"
GATE = ROOT / "formal_fixed_gate"
OUT = ROOT / "ab_curves"
SEEDS = [42, 43, 44]
COLORS = {"A": "#2ca02c", "B": "#d62728"}
PRETTY = {"A": "A (distance)", "B": "B (time)"}
SMOOTH = 20
UPDATES = list(range(20, 401, 20))


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
            update = int(row["update"])
            by_update[update] = row
    rows = [by_update[u] for u in sorted(by_update)]
    assert [int(r["update"]) for r in rows] == list(range(1, 401)), run
    return rows


def smooth(values, window=SMOOTH):
    array = np.asarray(values, dtype=float)
    if len(array) < window:
        return array
    kernel = np.ones(window) / window
    padded = np.pad(array, (window // 2, window - 1 - window // 2),
                    mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def validation_curve(run: str):
    rows = json.loads((V2 / "validation" / run / "validation_rows.json").read_text())
    by_update = {}
    for update in UPDATES:
        subset = [r for r in rows if r["checkpoint_update"] == update]
        completed = [r for r in subset if r["completed"]]
        scale_means = []
        for count in range(5, 21):
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

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex="col")
    for col, seed in enumerate(SEEDS):
        for objective in ("A", "B"):
            run = f"{objective}_seed{seed}"
            updates = np.array([int(r["update"]) for r in train[run]])
            reward = np.array([float(r["mean_ep_reward"]) for r in train[run]])
            critic = np.array([float(r["critic_loss"]) for r in train[run]])
            color = COLORS[objective]
            axes[0, col].plot(updates, reward, color=color, alpha=0.16, lw=0.7)
            axes[0, col].plot(updates, smooth(reward), color=color, lw=1.7,
                              label=PRETTY[objective])
            axes[1, col].plot(updates, critic, color=color, alpha=0.16, lw=0.7)
            axes[1, col].plot(updates, smooth(critic), color=color, lw=1.7)
        axes[0, col].set_title(f"seed {seed}", fontsize=11)
        axes[0, col].grid(True, alpha=0.25)
        axes[0, col].set_ylabel("mean completed-episode reward" if col == 0 else "")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("update")
        axes[1, col].set_ylabel("critic loss" if col == 0 else "")
        axes[1, col].grid(True, alpha=0.25, which="both")
        axes[1, col].set_xlim(1, 400)
    axes[0, 0].legend(loc="lower left", fontsize=8.5, frameon=True)
    fig.suptitle("A/B training curves (raw faint; 20-update moving average)", fontsize=15)
    fig.text(0.5, 0.935,
             "Reward and critic loss are objective-specific: A is distance reward, B is time reward; "
             "absolute values are not cross-objective comparable.",
             ha="center", fontsize=8.8, color="#4a5d4d")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "ab_training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ab_training_curves.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=False)
    for ax, seed in zip(axes, SEEDS):
        for objective in ("A", "B"):
            run = f"{objective}_seed{seed}"
            data = validation[run]
            updates = [u for u in UPDATES if data[u]["scale_equal_time"] is not None]
            values = [data[u]["scale_equal_time"] for u in updates]
            color = COLORS[objective]
            ax.plot(updates, values, color=color, marker="o", ms=3.5,
                    lw=1.4, label=PRETTY[objective])
            chosen = selected[run]
            ax.scatter([chosen], [data[chosen]["scale_equal_time"]], marker="*",
                       s=180, c=color, edgecolor="white", linewidth=0.8, zorder=6)
            ax.annotate(f"{chosen}", (chosen, data[chosen]["scale_equal_time"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7.2, color=color)
        ax.set_title(f"seed {seed}", fontsize=11)
        ax.set_xlabel("checkpoint update")
        ax.set_ylabel("scale-equal validation total time (s)" if seed == 42 else "")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 410)
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle("Validation time curves (96 fixed instances, all complete; lower is better)", fontsize=14)
    fig.text(0.5, 0.02,
             "Stars mark the checkpoint selected by the frozen completion-first rule. "
             "Time metric: model-estimated flight time with maneuver loss, excluding unmodeled service/landing.",
             ha="center", fontsize=8.2, color="#4a5d4d")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path2 = OUT / "ab_validation_time_curves.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "ab_validation_time_curves.svg", bbox_inches="tight")
    plt.close(fig)

    # Save the plotted values for auditability.
    payload = {"seeds": SEEDS, "smooth_window": SMOOTH,
               "training": {}, "validation": {}, "selected_update": selected}
    for run, rows in train.items():
        payload["training"][run] = [
            {"update": int(r["update"]),
             "mean_ep_reward": float(r["mean_ep_reward"]),
             "critic_loss": float(r["critic_loss"]),
             "explained_variance": float(r["explained_variance"]),
             "entropy": float(r["entropy"])}
            for r in rows
        ]
    for run, data in validation.items():
        payload["validation"][run] = {str(k): v for k, v in data.items()}
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
    print(path)
    print(path2)


if __name__ == "__main__":
    main()
