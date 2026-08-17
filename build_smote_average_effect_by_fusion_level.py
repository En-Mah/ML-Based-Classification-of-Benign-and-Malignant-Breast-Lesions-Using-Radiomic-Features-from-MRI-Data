#!/usr/bin/env python3
"""
Generate the grouped bar chart:
"Average effect of SMOTE by feature-fusion level"

Fixes in this version:
- extra headroom above the tallest bar
- smaller annotation offset
- title moved slightly farther from the axes
- value labels are kept inside the plotting area

Outputs:
    Figs/smote_average_effect_by_fusion_level.pdf
    Figs/smote_average_effect_by_fusion_level.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
LEVELS = ["Single", "Pairwise", "Triple", "Quadruple", "All five"]

# Average change after SMOTE = SMOTE - no-SMOTE
MEAN_OUTER_AUC_DELTA = np.array([
    0.0032,   # Single
    0.0015,   # Pairwise
   -0.0003,   # Triple
   -0.0006,   # Quadruple
   -0.0068,   # All five
])

POOLED_BALANCED_ACCURACY_DELTA = np.array([
    0.0155,   # Single
    0.0193,   # Pairwise
    0.0103,   # Triple
    0.0105,   # Quadruple
    0.0142,   # All five
])


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
OUTPUT_DIR = Path("Figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "smote_average_effect_by_fusion_level.pdf"
PNG_PATH = OUTPUT_DIR / "smote_average_effect_by_fusion_level.png"


# ---------------------------------------------------------------------
# Matplotlib settings
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
x = np.arange(len(LEVELS))
width = 0.36

fig, ax = plt.subplots(figsize=(10, 6))

bars_auc = ax.bar(
    x - width / 2,
    MEAN_OUTER_AUC_DELTA,
    width,
    label="Mean outer-fold ROC-AUC",
)

bars_bal = ax.bar(
    x + width / 2,
    POOLED_BALANCED_ACCURACY_DELTA,
    width,
    label="Pooled balanced accuracy",
)

# Zero reference line
ax.axhline(0, linewidth=1.0)

# ---------------------------------------------------------------------
# Labels / title
# ---------------------------------------------------------------------
ax.set_title(
    "Average effect of SMOTE by feature-fusion level",
    pad=14,   # more separation from the axes
)

ax.set_xlabel("Feature-fusion level")
ax.set_ylabel("Average change after SMOTE")

ax.set_xticks(x)
ax.set_xticklabels(LEVELS)

ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
ax.set_axisbelow(True)

ax.legend(loc="upper right", frameon=True)


# ---------------------------------------------------------------------
# Axis limits
# IMPORTANT:
# The previous upper limit (0.0206) was too close to the tallest bar
# (0.0193), so its +0.019 annotation entered the title area.
# ---------------------------------------------------------------------
Y_MIN = -0.0085
Y_MAX = 0.0228

ax.set_ylim(Y_MIN, Y_MAX)


# ---------------------------------------------------------------------
# Value annotations
# ---------------------------------------------------------------------
positive_offset = 0.00055
negative_offset = 0.00055

def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        xpos = bar.get_x() + bar.get_width() / 2

        if value >= 0:
            ypos = value + positive_offset
            va = "bottom"
        else:
            ypos = value - negative_offset
            va = "top"

        ax.text(
            xpos,
            ypos,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
            clip_on=True,   # prevents text from spilling into the title
        )


add_value_labels(bars_auc, MEAN_OUTER_AUC_DELTA)
add_value_labels(bars_bal, POOLED_BALANCED_ACCURACY_DELTA)


# ---------------------------------------------------------------------
# Layout and save
# ---------------------------------------------------------------------
# Leave explicit space for the title instead of relying only on tight_layout.
fig.subplots_adjust(
    left=0.10,
    right=0.97,
    bottom=0.12,
    top=0.88,
)

fig.savefig(PDF_PATH, bbox_inches="tight")
fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")

print(f"Saved PDF: {PDF_PATH}")
print(f"Saved PNG: {PNG_PATH}")

plt.show()
