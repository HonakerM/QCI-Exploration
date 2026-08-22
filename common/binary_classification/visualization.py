"""Create ROC, PR, metric, and timing charts for saved model results."""

from pathlib import Path
import re

from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .data_types import ModelResults


def _get_colors(n: int) -> list:
    """Build a palette with enough visually distinct colors for the models.

    Args:
        n (int): Number of colors needed.

    Returns:
        list: Color entries for plotting.
    """
    cmap = colormaps["tab20"] if n <= 20 else colormaps["hsv"]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_roc_curves(
    results: list[ModelResults],
    save_path: Path | None = None,
):
    """Plot ROC curves for each saved model run.

    Args:
        results (list[ModelResults]): Model results to compare.
        save_path (Path | None): Optional PNG path for saving the figure.
    """
    results = sorted(results, key=lambda r: r.auc, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    for r, color in zip(results, _get_colors(len(results))):
        ax.plot(
            r.fpr,
            r.tpr,
            linewidth=2,
            color=color,
            label=f"{r.run_label} (AUC = {r.auc:.4f}, AP = {getattr(r, 'auc_pr', 0.0):.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves: "
        + " vs ".join(r.run_label for r in results)
        + "\n Fraud Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_pr_curves(
    results: list[ModelResults],
    save_path: Path | None = None,
) -> None:
    """Plot precision-recall curves for each model result.

    Args:
        results (list[ModelResults]): Model results to compare.
        save_path (Path | None): Optional PNG path for saving the figure.
    """
    results = sorted(
        results, key=lambda r: r.auc_pr if hasattr(r, "auc_pr") else 0.0, reverse=True
    )
    fig, ax = plt.subplots(figsize=(10, 8))

    for r, color in zip(results, _get_colors(len(results))):
        precision = getattr(r, "pr_precision", None)
        recall = getattr(r, "pr_recall", None)
        if precision is None or recall is None or len(precision) == 0:
            continue
        ax.plot(
            recall,
            precision,
            linewidth=2,
            color=color,
            label=f"{r.run_label} (AP = {getattr(r, 'auc_pr', 0.0):.4f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves: "
        + " vs ".join(r.run_label for r in results)
        + "\n Fraud Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_metric_comparison(
    results: list[ModelResults],
    save_path: Path | None = None,
) -> None:
    """Compare model AUC and log loss with side-by-side bar charts.

    Args:
        results (list[ModelResults]): Model results to compare.
        save_path (Path | None): Optional PNG path for saving the figure.
    """
    names = [r.run_label for r in results]
    aucs = [r.auc for r in results]
    losses = [r.log_loss for r in results]
    colors = _get_colors(len(results))

    fig, (ax_auc, ax_loss) = plt.subplots(1, 2, figsize=(14, 6))

    _bar_chart(
        ax_auc,
        names,
        aucs,
        colors,
        ylabel="AUC Score",
        title="AUC-ROC Score Comparison",
        ylim=(0, 1.0),
        label_offset=0.02,
    )

    _bar_chart(
        ax_loss,
        names,
        losses,
        colors,
        ylabel="Log Loss",
        title="Log Loss Comparison (Lower is Better)",
        ylim=None,
        label_offset=max(losses) * 0.02,
    )

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_timing_comparison(
    results: list[ModelResults],
    save_path: Path | None = None,
) -> None:
    """Compare model runtime by stage in a stacked bar chart.

    Args:
        results (list[ModelResults]): Model results to compare.
        save_path (Path | None): Optional PNG path for saving the figure.
    """
    names = [r.run_label for r in results]
    timing_values = [r.timing for r in results]
    adapter_keys: list[str] = []
    seen_keys: set[str] = set()
    for timing in timing_values:
        for key in timing.adapter:
            if key not in seen_keys:
                seen_keys.add(key)
                adapter_keys.append(key)

    labels = ["Data Prep", "Fit", "Predict", *[key.title() for key in adapter_keys]]
    colors = _get_colors(len(labels))

    fig, ax = plt.subplots(figsize=(14, 7))

    spacing = 10
    bar_width = 0.8
    x_positions = [i * spacing for i in range(len(results))]
    bottoms = [0.0] * len(results)
    components = [
        [timing.data_prep for timing in timing_values],
        [timing.fit if timing.fit is not None else 0.0 for timing in timing_values],
        [timing.predict for timing in timing_values],
    ]
    components.extend(
        [
            [timing.adapter.get(key, 0.0) for timing in timing_values]
            for key in adapter_keys
        ]
    )

    totals = [sum(vals) for vals in zip(*components)]
    max_total = max(totals) if totals else 0.0
    # Segments shorter than this fraction of the tallest bar get pulled
    # into a stacked text block above the bar instead of an inline label.
    small_threshold = 0.03 * max_total if max_total else 0.0

    # Collect small-segment label lines per bar, in stacking order.
    pending_outside_labels: list[list[str]] = [[] for _ in results]

    for component_values, color, label in zip(components, colors, labels):
        bars = ax.bar(
            x_positions,
            component_values,
            width=bar_width,
            bottom=bottoms,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
            label=label,
        )
        for idx, value in enumerate(component_values):
            if value <= 0:
                continue
            seg_bottom = bottoms[idx]
            if value >= small_threshold:
                # Plenty of room: center the label inside the segment.
                ax.text(
                    bars[idx].get_x() + bars[idx].get_width() / 2,
                    seg_bottom + value / 2,
                    f"{value:.3f}s",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                # Too thin to hold text inline: defer to the stacked
                # text block drawn above the bar.
                pending_outside_labels[idx].append(f"{label}: {value:.3f}s")

        bottoms = [bottom + value for bottom, value in zip(bottoms, component_values)]

    # Draw one multi-line text block just above each bar (offset in points,
    # so it's independent of the x-axis scale/range) listing any segments
    # too small to label inline.
    for idx, lines in enumerate(pending_outside_labels):
        if not lines:
            continue
        ax.annotate(
            "\n".join(lines),
            xy=(x_positions[idx], bottoms[idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Give the stacked text blocks room to breathe above the tallest bar.
    max_lines = max((len(v) for v in pending_outside_labels), default=0)
    headroom = max_total * (0.06 * max(max_lines, 1) + 0.05) if max_total else 1.0
    ax.set_ylim(0, max_total + headroom)

    # Keep bars from ballooning to fill the figure width when there's only
    # one or two of them.
    ax.set_xlim(x_positions[0] - spacing / 2, x_positions[-1] + spacing / 2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Seconds", fontsize=12)
    ax.set_title("Timing Comparison by Training Stage", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _wrap_label(label: str, width: int) -> str:
    """Wrap a label into whole words on newlines

    Args:
        label (str): the label to split
        width (int): the max width of each line

    Returns:
        str: the new string
    """
    # Tokens are:
    #   - parenthesized groups: "(GPU)"
    #   - slash
    #   - non-whitespace sequences
    tokens = re.findall(r"\([^\s()]+\)|/|[^\s/]+", label)

    lines = []
    current = ""

    for token in tokens:
        if not current:
            current = token
        elif len(current) + 1 + len(token) <= width:
            current += " " + token
        else:
            lines.append(current)
            current = token

    if current:
        lines.append(current)

    return "\n".join(lines)


def _bar_chart(
    ax: Axes,
    names: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
    ylim: tuple | None,
    label_offset: float,
) -> None:
    """Draws a single labeled bar chart onto the given axes."""
    smallest_word_size = min(
        max(len(word) for name in names for word in re.split(r"[ ()/\\]", name)), 5
    )
    names = [_wrap_label(name, smallest_word_size) for name in names]
    ranked_tuples = sorted(zip(values, names, colors), reverse=True)
    values = [t[0] for t in ranked_tuples]
    names = [t[1] for t in ranked_tuples]
    colors = [t[2] for t in ranked_tuples]

    spacing = 10
    scaled_names = [i * spacing for i in range(len(names))]

    bars = ax.bar(
        scaled_names, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    ax.set_xticks(scaled_names)
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + label_offset,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def _save_or_show(fig: Figure, save_path: Path | None) -> None:
    """Saves the figure to disk, or shows it interactively.

    Args:
        fig (plt.Figure): The figure to save or display.
        save_path (Path | None): If provided, save the figure here as a PNG; otherwise
            call plt.show().

    Returns:
        None.
    """
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
