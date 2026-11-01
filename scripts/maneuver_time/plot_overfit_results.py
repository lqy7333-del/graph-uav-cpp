"""Plots and tables for the fixed 10x20 in-sample overfit diagnostic."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/run_seed42")
OUT = Path("/tmp/fixed-instance-overfit/docs/fixed_instance_overfit")
COL = {"initial": "#ff8c42", "nn": "#6b6b6b", "best": "#2e8b57",
       "final": "#2b6cb0"}
COMP = {"work": "#4c78a8", "nonwork": "#f58518", "maneuver": "#e45756"}


def rows(update):
    return json.loads((ROOT / f"evaluation/update_{update}.json").read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = json.loads((ROOT / "references.json").read_text())
    best = json.loads((ROOT / "best.json").read_text())
    curve = list(csv.DictReader((ROOT / "learning_curve.csv").open()))
    initial = refs["initial"]
    nn = refs["nn"]
    b = rows(best["update"])["rows"]
    f = rows(1000)["rows"]
    init_t = np.mean([r["total_time"] for r in initial])
    nn_t = np.mean([r["total_time"] for r in nn])
    best_t = np.mean([r["total_time"] for r in b])
    final_t = np.mean([r["total_time"] for r in f])

    # Learning curve, no smoothing.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = [int(r["update"]) for r in curve]
    y = [float(r["total_time"]) for r in curve]
    axes[0].plot(x, y, color="#1f77b4", marker="o", ms=3, lw=1.2,
                 label="greedy policy")
    axes[0].axhline(init_t, color=COL["initial"], ls="--", label=f"initial {init_t:.1f}s")
    axes[0].axhline(nn_t, color=COL["nn"], ls=":", label=f"NN {nn_t:.1f}s")
    axes[0].scatter([best["update"]], [best_t], marker="*", s=220,
                    color=COL["best"], edgecolor="white", zorder=5,
                    label=f"best update {best['update']}")
    axes[0].scatter([1000], [final_t], marker="s", s=80,
                    color=COL["final"], edgecolor="white", zorder=5,
                    label="final update 1000")
    axes[0].annotate(f"best {best_t:.1f}s\n+{best['initial_minus_candidate_total_time']:.1f}s vs initial\n+{best['nn_minus_candidate_total_time']:.1f}s vs NN",
                     (best["update"], best_t), xytext=(10, 20),
                     textcoords="offset points", fontsize=8, color=COL["best"])
    axes[0].annotate(
        f"final {final_t:.1f}s\n{init_t - final_t:+.1f}s vs initial\n"
        f"{nn_t - final_t:+.1f}s vs NN",
        (1000, final_t), xytext=(-135, 18), textcoords="offset points",
        fontsize=8, color=COL["final"])
    axes[0].set_ylabel("mean total time (s)")
    axes[0].set_title("Fixed 10-instance bank: in-sample greedy evaluation")
    axes[0].grid(alpha=.25)
    axes[0].legend(fontsize=7.6, loc="upper right")

    axes[1].plot(x, [float(r["initial_minus_candidate_total_time"]) for r in curve],
                 color=COL["initial"], marker="o", ms=2.5, lw=1.0,
                 label="initial - candidate")
    axes[1].plot(x, [float(r["nn_minus_candidate_total_time"]) for r in curve],
                 color=COL["nn"], marker="o", ms=2.5, lw=1.0,
                 label="NN - candidate")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_xlabel("fine-tuning update")
    axes[1].set_ylabel("time saved (s); positive is better")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=7.6)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Per-instance raw times and saved time.
    labels = [f"{r['index']}\n{r['seed']}" for r in initial]
    xx = np.arange(10)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    width = .20
    for offset, (name, rs) in zip((-1.5, -0.5, 0.5, 1.5),
                                   (("initial", initial), ("NN", nn),
                                   ("best", b), ("final", f))):
        vals = [r["total_time"] for r in rs]
        axes[0].bar(xx + offset * width, vals, width=width,
                    color=COL[name.lower()], label=name)
    axes[0].set_ylabel("total time (s)")
    axes[0].set_ylim(5500, 9000)
    axes[0].set_title("Per fixed instance: initial, NN, best, final")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    init_by = {r["index"]: r["total_time"] for r in initial}
    best_by = {r["index"]: r["total_time"] for r in b}
    final_by = {r["index"]: r["total_time"] for r in f}
    db = [init_by[i] - best_by[i] for i in range(10)]
    df = [init_by[i] - final_by[i] for i in range(10)]
    axes[1].bar(xx - .18, db, width=.34, color=COL["best"], label="initial - best")
    axes[1].bar(xx + .18, df, width=.34, color=COL["final"], label="initial - final")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_ylabel("time saved vs initial (s)")
    axes[1].set_xlabel("fixed instance index / seed")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "per_instance.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Mean time components.
    def means(rs):
        return {k: float(np.mean([r[k] for r in rs])) for k in
                ("time_work_base", "time_nonwork_base", "time_maneuver_extra")}
    names = ["initial", "best", "final"]
    values = [means(initial), means(b), means(f)]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bottoms = np.zeros(3)
    for key, label in (("time_work_base", "work"),
                       ("time_nonwork_base", "nonwork"),
                       ("time_maneuver_extra", "maneuver")):
        vals = np.array([v[key] for v in values])
        ax.bar(names, vals, bottom=bottoms, color=COMP[label], label=label, width=.58)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(i, total + 25, f"{total:.1f}s", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "time_components.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # Machine-readable per-instance table.
    with (OUT / "per_instance.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "seed", "initial_time", "nn_time", "best_time",
                         "final_time", "initial_minus_best", "nn_minus_best",
                         "initial_minus_final", "nn_minus_final"])
        for i in range(10):
            writer.writerow([i, initial[i]["seed"], initial[i]["total_time"],
                             nn[i]["total_time"], b[i]["total_time"],
                             f[i]["total_time"],
                             initial[i]["total_time"] - b[i]["total_time"],
                             nn[i]["total_time"] - b[i]["total_time"],
                             initial[i]["total_time"] - f[i]["total_time"],
                             nn[i]["total_time"] - f[i]["total_time"]])

    summary = {
        "initial": init_t, "nn": nn_t, "best": best_t, "final": final_t,
        "best_update": best["update"],
        "best_initial_gain": best["initial_minus_candidate_total_time"],
        "best_nn_gain": best["nn_minus_candidate_total_time"],
        "final_initial_gain": init_t - final_t,
        "final_nn_gain": nn_t - final_t,
        "best_initial_wins": best["initial_wins"],
        "best_initial_losses": best["initial_losses"],
        "best_nn_wins": best["nn_wins"],
        "best_nn_losses": best["nn_losses"],
        "final_initial_wins": sum(init_by[i] - final_by[i] > 1e-6 for i in range(10)),
        "final_initial_losses": sum(init_by[i] - final_by[i] < -1e-6 for i in range(10)),
        "final_nn_wins": sum(nn[i]["total_time"] - final_by[i] > 1e-6 for i in range(10)),
        "final_nn_losses": sum(nn[i]["total_time"] - final_by[i] < -1e-6 for i in range(10)),
        "best_gain_sum": sum(db),
        "top3_gain": sum(sorted(db, reverse=True)[:3]),
        "best_gain_max": max(db),
        "best_gain_max_index": int(np.argmax(db)),
        "components_initial": means(initial), "components_best": means(b),
        "components_final": means(f),
    }
    (OUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
