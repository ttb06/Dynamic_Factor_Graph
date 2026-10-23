#!/usr/bin/env python3
"""Visualize verified subgroup graphs and their activation along diffusion.

The script consumes aggregate artifacts only; it neither loads nor emits source
rows.  Edges are empirical hypotheses, not asserted domain invariants.
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


TIME_ORDER = ["(0.75,1.0]", "(0.50,0.75]", "(0.25,0.50]", "(0.10,0.25]", "[0,0.10]"]
TIME_LABEL = ["early\n(>.75)", ".50–.75", ".25–.50", ".10–.25", "late\n(≤.10)"]
COLORS = {"edge_active_in_subgroup": "#16865C", "exception_subgroup": "#D97706"}


def relation_parts(text: str) -> tuple[str, str]:
    match = re.match(r"IF\s+(.*?)\s+THEN\s+(.*)", str(text))
    return match.groups() if match else (str(text), "")


def short(text: str, length: int = 25) -> str:
    text = str(text).replace("abs_", "abs(")
    return text if len(text) <= length else text[: length - 1] + "…"


def node_label(text: str) -> str:
    return textwrap.fill(str(text), width=18, break_long_words=False)


def select_edges(frame: pd.DataFrame, dataset: str, per_kind: int = 2) -> pd.DataFrame:
    part = frame[
        frame.dataset.eq(dataset)
        & ~frame.sensitive_or_demographic.astype(bool)
        & ~frame.target_related.astype(bool)
    ].copy()
    part["strength"] = part.test_interaction.abs()
    selected = []
    for _, block in part.groupby("graph_interpretation"):
        selected.append(block.sort_values(["strength", "test_group_support"], ascending=False).head(per_kind))
    if not selected:
        return part.head(0)
    return pd.concat(selected).sort_values("strength", ascending=False).head(4)


def graph_panels(edges: pd.DataFrame, output: Path) -> pd.DataFrame:
    # Avoid displaying health-code associations in the default report even if
    # a coarse sensitive-column detector did not flag them.
    datasets = [name for name in ["default", "news", "beijing"] if edges.dataset.eq(name).any()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 7.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    chosen = []
    for ax, dataset in zip(axes.flat, datasets):
        block = select_edges(edges, dataset)
        chosen.append(block)
        ax.set_title(dataset.upper(), loc="left", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, max(len(block), 1) - 0.5)
        ax.axis("off")
        for row_index, row in enumerate(block.itertuples(index=False)):
            y = len(block) - row_index - 1
            antecedent, consequent = relation_parts(row.constraint)
            color = COLORS[row.graph_interpretation]
            for x, label in [(0.08, antecedent), (0.72, consequent)]:
                box = FancyBboxPatch(
                    (x, y - 0.18), 0.21, 0.36,
                    boxstyle="round,pad=0.02,rounding_size=0.025",
                    facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.1,
                )
                ax.add_patch(box)
                ax.text(x + 0.105, y, node_label(label), ha="center", va="center", fontsize=7.3)
            arrow = FancyArrowPatch(
                (0.30, y), (0.71, y), arrowstyle="-|>", mutation_scale=13,
                linewidth=2.2, color=color,
            )
            ax.add_patch(arrow)
            scope = short(row.subgroup, 25)
            rates = f"heldout violation: {row.test_group_violation:.2f} vs {row.test_other_violation:.2f}"
            ax.text(0.505, y + 0.15, f"scope: {scope}", ha="center", va="bottom", fontsize=7.7, color=color)
            ax.text(0.505, y - 0.15, rates, ha="center", va="top", fontsize=7.2, color="#475569")
    fig.legend(
        handles=[
            Line2D([0], [0], color=COLORS["edge_active_in_subgroup"], lw=3, label="more reliable inside subgroup"),
            Line2D([0], [0], color=COLORS["exception_subgroup"], lw=3, label="exception-prone inside subgroup"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=10,
    )
    fig.suptitle("Heldout-replicated subgroup-conditioned graph edges", fontsize=17, fontweight="bold")
    fig.savefig(output / "dataset_subgroup_graphs.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "dataset_subgroup_graphs.svg", bbox_inches="tight")
    plt.close(fig)
    selected = pd.concat(chosen, ignore_index=True)
    selected.to_csv(output / "selected_dataset_edges.csv", index=False)
    return selected


def trajectory_heatmaps(trajectory: pd.DataFrame, output: Path) -> pd.DataFrame:
    scoped = trajectory[trajectory.scope.eq("edge_scope")].copy()
    summary = scoped.groupby(
        ["candidate_id", "constraint", "subgroup", "time_bin"], as_index=False
    ).agg(
        active_fraction=("predicted_positive_rate", "mean"),
        precision=("precision_to_final", "mean"),
        recall=("recall_to_final", "mean"),
        accuracy=("accuracy_to_final", "mean"),
        flip_rate=("flip_from_previous", "mean"),
        final_fraction=("final_positive_rate", "mean"),
    )
    candidates = (
        summary.groupby(["candidate_id", "constraint", "subgroup"], as_index=False)
        .agg(final_fraction=("final_fraction", "mean"))
        .sort_values("final_fraction", ascending=False)
        .head(8)
    )
    keys = candidates.candidate_id.tolist()
    view = summary[summary.candidate_id.isin(keys)].copy()
    labels = []
    for row in candidates.itertuples(index=False):
        a, c = relation_parts(row.constraint)
        labels.append(f"{short(a, 18)} → {short(c, 18)}\nwhen {short(row.subgroup, 19)}")

    def matrix(column: str) -> np.ndarray:
        pivot = view.pivot(index="candidate_id", columns="time_bin", values=column)
        return pivot.reindex(index=keys, columns=TIME_ORDER).to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), constrained_layout=True)
    specs = [
        ("active_fraction", "Fraction of rows with active edge", "Blues", 0, 1),
        ("precision", "Precision against final active set", "Greens", 0, 1),
        ("recall", "Recall against final active set", "Oranges", 0, 1),
    ]
    for ax, (column, title, cmap, vmin, vmax) in zip(axes, specs):
        values = matrix(column)
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(TIME_LABEL)), TIME_LABEL, fontsize=9)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [""] * len(labels), fontsize=8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if values[i, j] > 0.58 else "#172033")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Default: edge activation becomes reliable along reverse diffusion", fontsize=17, fontweight="bold")
    fig.savefig(output / "default_edge_activation_by_timestep.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "default_edge_activation_by_timestep.svg", bbox_inches="tight")
    plt.close(fig)
    view.to_csv(output / "trajectory_edge_activation_summary.csv", index=False)
    return view


def write_report(selected: pd.DataFrame, trajectory: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Dynamic graph visual audit",
        "",
        "These figures use aggregate outputs only. The discovered edges are heldout-replicated empirical hypotheses, not certified domain truths.",
        "",
        "## Figures",
        "",
        "- `dataset_subgroup_graphs.png`: selected edges for Default, News and Beijing. Green means the edge is more reliable inside the displayed subgroup; orange means the subgroup is exception-prone.",
        "- `default_edge_activation_by_timestep.png`: for eight Default edges, the fraction of synthetic rows activating each edge and its precision/recall relative to final-row membership.",
        "",
        "## Selected dataset edges",
        "",
        "| Dataset | Relation | Subgroup | Interpretation | Heldout violation in/out |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.constraint} | {row.subgroup} | {row.graph_interpretation} | "
            f"{row.test_group_violation:.3f}/{row.test_other_violation:.3f} |"
        )
    bins = trajectory.groupby("time_bin", as_index=False).agg(
        active_fraction=("active_fraction", "mean"), precision=("precision", "mean"), recall=("recall", "mean")
    ).set_index("time_bin").reindex(TIME_ORDER).reset_index()
    lines += [
        "",
        "## Mean activation quality across displayed Default edges",
        "",
        "| Timestep bin | Active fraction | Precision | Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in bins.itertuples(index=False):
        lines.append(f"| {row.time_bin} | {row.active_fraction:.3f} | {row.precision:.3f} | {row.recall:.3f} |")
    lines += [
        "",
        "## Reading guide",
        "",
        "- An orange edge should normally be inactive inside its displayed subgroup, or handled as abstain until an independent semantic verifier approves it.",
        "- A green edge is a candidate for activation inside its subgroup, but statistical replication alone does not justify hard enforcement.",
        "- `active_fraction` is aggregate row-level scope membership, not the probability that the constraint is semantically true.",
        "- The trajectory comparison uses final generated-row membership as its reference label; it measures state readability rather than external semantic correctness.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    edges = pd.read_csv(args.edges)
    trajectory = pd.read_csv(args.trajectory)
    selected = graph_panels(edges, args.output)
    trajectory_summary = trajectory_heatmaps(trajectory, args.output)
    write_report(selected, trajectory_summary, args.output)


if __name__ == "__main__":
    main()
