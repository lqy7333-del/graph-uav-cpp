"""Plot the bounded sampled-vs-greedy diagnostic and export rank/instance usage."""
from __future__ import annotations
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/run_seed42")
OUT = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/sampled_diagnostic")
DOC = Path("/tmp/fixed-instance-overfit/docs/fixed_instance_overfit")
CHECKPOINTS = [0, 460, 580, 600, 780, 1000]


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    d = json.loads((OUT / "summary.json").read_text())
    x = np.array(CHECKPOINTS)
    greedy = np.array([d["results"][str(u)]["greedy_mean"] for u in CHECKPOINTS])
    sampled = np.array([d["results"][str(u)]["sampled_mean"] for u in CHECKPOINTS])
    low = np.array([d["results"][str(u)]["sampled_ci95"][0] for u in CHECKPOINTS])
    high = np.array([d["results"][str(u)]["sampled_ci95"][1] for u in CHECKPOINTS])
    delta = sampled - greedy

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    axes[0].errorbar(x, sampled, yerr=[sampled - low, high - sampled],
                     fmt="o-", color="#d95f02", capsize=5, lw=1.4,
                     label="sampled mean (16 per instance, 95% bootstrap CI)")
    axes[0].plot(x, greedy, "s-", color="#1b9e77", lw=1.4,
                 label="greedy evaluation")
    axes[0].set_ylabel("mean total time (s)")
    axes[0].set_title("Same 10 fixed instances: sampled vs greedy")
    axes[0].grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[1].bar(range(len(x)), delta, color="#7570b3", width=.55)
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_xticks(range(len(x)), [str(u) for u in x])
    axes[1].set_xlim(-0.5, len(x) - 0.5)
    axes[1].set_ylabel("sampled − greedy (s)")
    axes[1].set_xlabel("fine-tuning checkpoint")
    axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(DOC / "sampled_vs_greedy.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Per-instance sampled - greedy heatmap.
    matrix = np.array([[d["results"][str(u)]["per_instance_sampled_mean"][str(i)]
                        - d["results"][str(u)]["per_instance_greedy"][str(i)]
                        for i in range(10)] for u in CHECKPOINTS])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vmax = np.abs(matrix).max()
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(10), [f"{i}\n{70000+i}" for i in range(10)], fontsize=7)
    ax.set_yticks(range(len(CHECKPOINTS)), [str(u) for u in CHECKPOINTS])
    ax.set_ylabel("checkpoint update")
    ax.set_xlabel("fixed instance index / seed")
    ax.set_title("Per-instance sampled mean − greedy (s); positive = sampled slower")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.0f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    fig.colorbar(image, ax=ax, label="s")
    fig.tight_layout()
    fig.savefig(DOC / "sampled_per_instance_delta.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Per-rank/per-instance usage at the selected checkpoints.
    rows = []
    for update in CHECKPOINTS:
        payload = json.loads((RUN / f"evaluation/update_{update}.json").read_text())
        for rank, usage in enumerate(payload["rank_usage"]):
            for index in range(10):
                rows.append({
                    "update": update, "rank": rank, "index": index,
                    "seed": 70000 + index,
                    "starts": usage["starts"][index],
                    "transitions": usage["transitions"][index],
                    "completed_episodes": usage["completed_episodes"][index],
                })
    with (DOC / "rank_instance_usage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(DOC / "sampled_vs_greedy.png")
    print(DOC / "sampled_per_instance_delta.png")


if __name__ == "__main__":
    main()
