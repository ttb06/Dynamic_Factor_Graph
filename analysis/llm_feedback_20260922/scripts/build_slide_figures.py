#!/usr/bin/env python3
"""Generate compact, slide-ready figures from executed experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def horizontal_bar(filename, labels, values, xlabel, colors, xlim):
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel(xlabel); ax.set_xlim(*xlim); ax.grid(axis="x", alpha=.20)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + (xlim[1] * .018), bar.get_y() + bar.get_height()/2,
                f"{value:.2f}%", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=200); plt.close(fig)


horizontal_bar(
    "full_checkpoint_baselines.png",
    ["No guidance", "Late once", "Targeted recheck", "Fixed each step", "Post-repair"],
    [1.758, .195, .130, 0, 0],
    "Hard-rule violation rate (%)",
    ["#374F72", "#1C978B", "#1C978B", "#3E9761", "#EE8B3D"],
    (0, 1.98),
)

horizontal_bar(
    "actuator_comparison.png",
    ["No guidance", "x0 correction @ .03", "State projection @ .03",
     "State projection @ .10", "State recheck .10→.03", "Post-repair"],
    [1.5625, .2604, .1302, .0651, .0651, 0],
    "Final hard-rule violation rate (%)",
    ["#374F72", "#CE4949", "#1C978B", "#1C978B", "#1C978B", "#EE8B3D"],
    (0, 1.78),
)

names = ["Qwen\ntyped", "Qwen\nrubric", "Qwen\ntemporal",
         "GPT-4.1-mini\nfocused", "GPT-5.4-mini\nfocused"]
precision = [.352, .185, .525, .989, .978]
recall = [.190, .200, .210, .870, .870]
x = np.arange(len(names)); width = .34
fig, ax = plt.subplots(figsize=(9.0, 4.4))
ax.bar(x-width/2, precision, width, label="Precision", color="#1C978B")
ax.bar(x+width/2, recall, width, label="Recall", color="#EE8B3D")
ax.set_xticks(x, names); ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
ax.grid(axis="y", alpha=.20); ax.legend(frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "open_vs_focused_llm.png", dpi=200); plt.close(fig)

relations = ["Education FD", "Numeric domain", "Spouse", "Age–education", "Missingness"]
values = [1, 1, 1, 1, .35]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
bars = ax.bar(relations, values, color=["#1C978B"]*4 + ["#EE8B3D"])
ax.set_ylim(0, 1.08); ax.set_ylabel("Recall"); ax.grid(axis="y", alpha=.20)
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, value+.025, f"{value:.2f}", ha="center")
fig.tight_layout(); fig.savefig(OUT / "focused_relation_recall.png", dpi=200); plt.close(fig)

print(OUT)
