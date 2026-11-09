"""Plot training curves and final results of all trained models.

Usage:
    python scripts/plot_training_curves.py [--output docs/training_curves.png]

Panels (data from the TensorBoard logs in ``artifacts/runs`` and ``artifacts/results.json``):
    1. validation NDCG@10 per epoch, all models;
    2. the same, zoomed in on the non-sequential models;
    3. training loss relative to epoch 1 (the models optimise different losses, so raw values
       are not comparable; indexing each to its first epoch shows convergence speed);
    4. test NDCG@10 of the selected checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recsys.config import DEFAULT_ARTIFACTS_DIR, MODEL_ORDER, results_path
from recsys.models import MODEL_REGISTRY

# Fixed categorical order (one hue per model, never cycled); the baseline stays neutral.
COLORS = {"popular": "#8a8984", "mf": "#2a78d6", "two_tower": "#eb6834", "sasrec": "#1baf7a"}
TRAINED = ("mf", "two_tower", "sasrec")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

Series = tuple[list[int], list[float]]


def read_scalars(run_dir: Path) -> dict[str, Series]:
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        scalars[tag] = ([e.step for e in events], [e.value for e in events])
    return scalars


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def legend(ax: plt.Axes, loc: str) -> None:
    for text in ax.legend(frameon=False, loc=loc).get_texts():
        text.set_color(INK)


def plot_validation(
    ax: plt.Axes, runs: dict[str, dict[str, Series]], names: tuple[str, ...], popular: float | None
) -> None:
    for name in names:
        steps, values = runs[name]["valid/NDCG@10"]
        best = max(range(len(values)), key=values.__getitem__)
        label = f"{MODEL_REGISTRY[name].display_name} (best {values[best]:.3f})"
        ax.plot(steps, values, color=COLORS[name], linewidth=2, label=label)
        ax.plot(
            steps[best],
            values[best],
            "o",
            color=COLORS[name],
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    if popular is not None:
        ax.axhline(popular, color=COLORS["popular"], linewidth=1.5, linestyle="--")
        ax.annotate(
            f"Most Popular {popular:.3f}",
            (1.0, popular),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            color=MUTED,
            fontsize=9,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=Path("docs") / "training_curves.png")
    args = parser.parse_args()

    runs = {
        name: read_scalars(args.artifacts_dir / "runs" / name)
        for name in TRAINED
        if (args.artifacts_dir / "runs" / name).exists()
    }
    if not runs:
        raise SystemExit("No TensorBoard runs found. Train models first.")
    results_file = results_path(args.artifacts_dir)
    results = (
        json.loads(results_file.read_text(encoding="utf-8"))["results"]
        if results_file.exists()
        else {}
    )
    popular = results["popular"]["valid"]["NDCG@10"] if "popular" in results else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), facecolor=SURFACE)
    (ax_all, ax_zoom), (ax_loss, ax_test) = axes

    plot_validation(ax_all, runs, tuple(runs), popular)
    style_axis(ax_all, "Validation NDCG@10 (dot = best epoch)", "epoch", "NDCG@10")
    ax_all.set_ylim(bottom=0)
    legend(ax_all, "center right")

    non_sequential = tuple(n for n in ("mf", "two_tower") if n in runs)
    plot_validation(ax_zoom, runs, non_sequential, popular)
    style_axis(ax_zoom, "Zoom: non-sequential models", "epoch", "NDCG@10")
    legend(ax_zoom, "upper left")

    for name, scalars in runs.items():
        steps, losses = scalars["train/loss"]
        ax_loss.plot(
            steps,
            [v / losses[0] for v in losses],
            color=COLORS[name],
            linewidth=2,
            label=MODEL_REGISTRY[name].display_name,
        )
    style_axis(ax_loss, "Training loss, relative to epoch 1", "epoch", "loss / loss at epoch 1")
    legend(ax_loss, "center right")

    tested = [n for n in MODEL_ORDER if n in results]
    if tested:
        values = [results[n]["test"]["NDCG@10"] for n in tested]
        positions = range(len(tested))
        ax_test.bar(positions, values, width=0.6, color=[COLORS[n] for n in tested])
        for x, value in zip(positions, values, strict=True):
            ax_test.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=10,
            )
        ax_test.set_xticks(list(positions), [MODEL_REGISTRY[n].display_name for n in tested])
        ax_test.set_ylim(0, max(values) * 1.15)
    style_axis(ax_test, "Test NDCG@10 (full ranking)", "", "NDCG@10")
    ax_test.grid(False, axis="x")

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
