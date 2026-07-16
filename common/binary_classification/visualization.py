"""Plotting helpers consumed by both fraud scripts."""

from pathlib import Path
import re

from matplotlib import colormaps
import matplotlib.pyplot as plt

from .data_types import ModelResults


def _get_colors(n: int) -> list:
    """Returns n visually distinct colors."""
    cmap = colormaps["tab20"] if n <= 20 else colormaps["hsv"]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_roc_curves(
    results: list[ModelResults],
    save_path: Path | None = None,
):
    """Plots overlaid ROC curves for every ModelResults in results.

    Args:
        results (list[ModelResults]): One ModelResults per model.
        save_path (Path | None): If provided, write the plot as a PNG here instead of
            calling plt.show().
    """
    results = sorted(results, key=lambda r: r.auc, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    for r, color in zip(results, _get_colors(len(results))):
        ax.plot(
            r.fpr,
            r.tpr,
            linewidth=2,
            color=color,
            label=f"{r.model_name} (AUC = {r.auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves: "
        + " vs ".join(r.model_name for r in results)
        + "\n Fraud Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_metric_comparison(
    results: list[ModelResults],
    save_path: Path | None = None,
) -> None:
    """Plots side-by-side bar charts of AUC and log loss.

    AUC is higher-is-better and log loss is lower-is-better.

    Args:
        results (list[ModelResults]): One ModelResults per model.
        save_path (Path | None): If provided, write the plot as a PNG here instead of
            calling plt.show().
    """
    names = [r.model_name for r in results]
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
    ax: plt.Axes,
    names: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
    ylim: tuple | None,
    label_offset: float,
) -> None:
    """Draws a single labeled bar chart onto the given axes.

    Args:
        ax (plt.Axes): Matplotlib axes to draw onto.
        names (list[str]): Bar labels, one per model.
        values (list[float]): Bar heights, one per model.
        colors (list[str]): Bar colors, one per model.
        ylabel (str): Y-axis label.
        title (str): Chart title.
        ylim (tuple | None): Optional (min, max) y-axis limits.
        label_offset (float): Vertical offset used to place the value label above
            each bar.
    """
    smallest_word_size = min(
        max(len(word) for name in names for word in re.split(r"[ ()/\\]", name)), 5
    )
    names = [_wrap_label(name, smallest_word_size) for name in names]
    ranked_tuples = sorted(zip(values, names, colors), reverse=True)
    values, names, colors = zip(*ranked_tuples)

    spacing = 3
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


def _save_or_show(fig: plt.Figure, save_path: Path | None) -> None:
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
