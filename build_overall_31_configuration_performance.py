#!/usr/bin/env python3
"""
Build a publication-quality 5-panel figure comparing the best pipeline for
all 31 radiomic source configurations:

(a) 5 single-source
(b) 10 pairwise
(c) 10 triple
(d) 5 quadruple
(e) best configuration at each fusion level, including all-five

The script reads the final publication-summary CSV files and uses:
    Mean_AUC  -> bar length
    Std_AUC   -> error bar

Outputs:
    overall_31_configuration_performance.pdf   (vector; recommended for LaTeX)
    overall_31_configuration_performance.svg   (vector)
    overall_31_configuration_performance.png   (600 dpi)

Example
-------
python build_overall_31_configuration_performance.py \
    --single Single/.../publication_single_dataset_summary.csv \
    --pairwise Pairwise/.../publication_all_10_pairwise_concat_summary.csv \
    --triple Triple/.../publication_all_10_triple_concat_summary.csv \
    --quadruple Quadruple/.../publication_all_5_quad_concat_summary.csv \
    --all-five AllFive/.../publication_all_5_concat_summary.csv \
    --output-dir Figs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------
XMIN = 0.50
XMAX = 0.90
XTICKS = np.arange(0.50, 0.91, 0.10)

# Colors correspond to fusion level, matching the requested figure.
COLORS = {
    "Single": "#1478C8",
    "Pairwise": "#18A6A1",
    "Triple": "#49A91E",
    "Quadruple": "#FF7900",
    "All-five": "#7A52B3",
}

EXPECTED_COUNTS = {
    "Single": 5,
    "Pairwise": 10,
    "Triple": 10,
    "Quadruple": 5,
    "All-five": 1,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Build the unified 31-configuration ROC-AUC figure."
    )
    p.add_argument("--single", required=True, type=Path)
    p.add_argument("--pairwise", required=True, type=Path)
    p.add_argument("--triple", required=True, type=Path)
    p.add_argument("--quadruple", required=True, type=Path)
    p.add_argument("--all-five", required=True, type=Path)
    p.add_argument("--output-dir", default=Path("Figs"), type=Path)
    p.add_argument(
        "--prefix",
        default="overall_31_configuration_performance",
        help="Output filename prefix.",
    )
    p.add_argument(
        "--no-count-check",
        action="store_true",
        help="Do not require exactly 5/10/10/5/1 source configurations.",
    )
    return p.parse_args()


def clean_col_name(name: str) -> str:
    """Normalize a column name for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, candidates, required=True):
    """
    Find the first matching column, case/punctuation insensitive.
    """
    normalized = {clean_col_name(c): c for c in df.columns}

    for candidate in candidates:
        key = clean_col_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            "Could not find any of these columns:\n"
            f"  {candidates}\n"
            f"Available columns:\n  {list(df.columns)}"
        )
    return None


def numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert a result column to numeric.

    Handles ordinary numeric values and strings such as:
        '0.836'
        '0.836 ± 0.088'
    In the latter case, the first numeric token is taken.
    """
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == len(s):
        return out

    extracted = (
        s.astype(str)
        .str.extract(r"([-+]?(?:\d*\.\d+|\d+))", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def canonicalize_label(label: str) -> str:
    """
    Keep labels compact and consistent for the manuscript figure.
    """
    x = str(label).strip()

    # Remove common dataset filename suffixes if they appear.
    x = re.sub(r"_M1(?:_(?:single|pair|pairwise|triple|quad|quadruple|all5))?$",
               "", x, flags=re.I)
    x = re.sub(r"\.csv$", "", x, flags=re.I)

    # Remove generic concat suffixes.
    x = re.sub(r"_(?:single|pairwise|pair|triple|quadruple|quad|all[_-]?five|concat)$",
               "", x, flags=re.I)

    return x


def load_level(path: Path, level: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{level} CSV not found:\n{path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{level} CSV is empty:\n{path}")

    mean_col = find_column(
        df,
        [
            "Mean_AUC",
            "Mean_ROC_AUC",
            "Mean_Outer_AUC",
            "Mean_Outer_ROC_AUC",
            "Outer_Mean_AUC",
        ],
    )

    std_col = find_column(
        df,
        [
            "Std_AUC",
            "SD_AUC",
            "AUC_SD",
            "Std_ROC_AUC",
            "SD_ROC_AUC",
            "Outer_AUC_SD",
            "Mean_AUC_SD",
        ],
    )

    if level == "Single":
        label_candidates = [
            "Single_Tag", "Dataset", "Dataset_Tag", "Source",
            "Configuration", "Config", "Name"
        ]
    elif level == "Pairwise":
        label_candidates = [
            "Pair_Tag", "Pair", "Dataset", "Dataset_Tag",
            "Configuration", "Config", "Name"
        ]
    elif level == "Triple":
        label_candidates = [
            "Triple_Tag", "Triple", "Dataset", "Dataset_Tag",
            "Configuration", "Config", "Name"
        ]
    elif level == "Quadruple":
        label_candidates = [
            "Quad_Tag", "Quadruple_Tag", "Quad", "Dataset", "Dataset_Tag",
            "Configuration", "Config", "Name"
        ]
    else:
        label_candidates = [
            "AllFive_Tag", "All_Five_Tag", "All5_Tag", "Concat_Tag",
            "Dataset", "Dataset_Tag", "Configuration", "Config", "Name"
        ]

    label_col = find_column(df, label_candidates, required=False)

    work = pd.DataFrame()
    if label_col is None:
        if level == "All-five":
            work["Configuration"] = ["Pre_Post1_Post2_ADC_T2"] * len(df)
        else:
            raise KeyError(
                f"Could not identify the configuration-name column for {level}.\n"
                f"Available columns: {list(df.columns)}"
            )
    else:
        work["Configuration"] = df[label_col].map(canonicalize_label)

    work["Mean_AUC"] = numeric_series(df[mean_col])
    work["Std_AUC"] = numeric_series(df[std_col])

    work = work.dropna(subset=["Configuration", "Mean_AUC", "Std_AUC"]).copy()

    # If a source configuration appears more than once, retain the row with
    # the highest mean AUC. This makes the script safe even if the supplied
    # table contains more than one candidate pipeline per source.
    work = (
        work.sort_values("Mean_AUC", ascending=False)
            .drop_duplicates(subset="Configuration", keep="first")
            .reset_index(drop=True)
    )

    work["Level"] = level
    return work


def validate_counts(data, do_check=True):
    if not do_check:
        return

    errors = []
    for level, df in data.items():
        expected = EXPECTED_COUNTS[level]
        observed = len(df)
        if observed != expected:
            errors.append(f"{level}: expected {expected}, found {observed}")

    if errors:
        raise ValueError(
            "Unexpected number of unique source configurations:\n  "
            + "\n  ".join(errors)
            + "\n\nUse the final publication-summary CSVs, or rerun with "
              "--no-count-check if the difference is intentional."
        )


def style_axis(ax, title):
    ax.set_xlim(XMIN - 0.005, XMAX)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Mean ROC-AUC", fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=7)

    # Chance-level discrimination.
    ax.axvline(
        XMIN,
        color="0.60",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=0,
    )

    ax.grid(False)
    ax.tick_params(axis="both", labelsize=8.5, width=0.8, length=3.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def plot_level(ax, df, level, title, mark_best=True):
    d = df.sort_values("Mean_AUC", ascending=False).reset_index(drop=True)

    y = np.arange(len(d))
    means = d["Mean_AUC"].to_numpy(float)
    stds = d["Std_AUC"].to_numpy(float)
    labels = d["Configuration"].astype(str).tolist()

    # Draw bars from chance level (0.50) to the observed mean AUC.
    widths = np.maximum(means - XMIN, 0)

    ax.barh(
        y,
        widths,
        left=XMIN,
        xerr=stds,
        height=0.66 if len(d) <= 5 else 0.62,
        color=COLORS[level],
        edgecolor="none",
        error_kw=dict(
            ecolor="black",
            elinewidth=0.85,
            capsize=2.7,
            capthick=0.85,
        ),
        zorder=2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    style_axis(ax, title)

    # Highlight only the best configuration for this fusion level.
    if len(d) and mark_best:
        ticklabels = ax.get_yticklabels()
        ticklabels[0].set_fontweight("bold")

        star_x = min(XMAX - 0.010, means[0] + stds[0] + 0.010)
        ax.text(
            star_x,
            0,
            "★",
            ha="center",
            va="center",
            fontsize=10.5,
            color="black",
            zorder=5,
        )


def make_panel_e(data):
    rows = []
    order = ["Single", "Pairwise", "Triple", "Quadruple", "All-five"]

    for level in order:
        d = data[level].sort_values("Mean_AUC", ascending=False).iloc[0]
        rows.append(
            {
                "Level": level,
                "Configuration": d["Configuration"],
                "Mean_AUC": float(d["Mean_AUC"]),
                "Std_AUC": float(d["Std_AUC"]),
            }
        )

    return pd.DataFrame(rows)


def plot_best_by_level(ax, best_df):
    y = np.arange(len(best_df))

    for i, row in best_df.iterrows():
        mean = float(row["Mean_AUC"])
        sd = float(row["Std_AUC"])
        level = row["Level"]

        ax.barh(
            i,
            max(mean - XMIN, 0),
            left=XMIN,
            xerr=sd,
            height=0.55,
            color=COLORS[level],
            edgecolor="none",
            error_kw=dict(
                ecolor="black",
                elinewidth=0.85,
                capsize=2.7,
                capthick=0.85,
            ),
            zorder=2,
        )

    labels = [
        f"{row.Level} ({row.Configuration})"
        for row in best_df.itertuples(index=False)
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    style_axis(ax, "(e) Best configuration at each fusion level")

    # Bold only the fusion-level word, approximately, by making all labels
    # semibold. This remains robust in PDF/SVG export.
    for lab in ax.get_yticklabels():
        lab.set_fontweight("semibold")


def main():
    args = parse_args()

    data = {
        "Single": load_level(args.single, "Single"),
        "Pairwise": load_level(args.pairwise, "Pairwise"),
        "Triple": load_level(args.triple, "Triple"),
        "Quadruple": load_level(args.quadruple, "Quadruple"),
        "All-five": load_level(args.all_five, "All-five"),
    }

    validate_counts(data, do_check=not args.no_count_check)

    # 12 x 9 inches = exact 4:3 aspect ratio.
    fig = plt.figure(figsize=(12, 9), constrained_layout=False)

    # Bottom panel is slightly shorter than the two upper rows.
    gs = GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        height_ratios=[1.00, 1.00, 0.78],
        hspace=0.48,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    plot_level(
        ax_a,
        data["Single"],
        "Single",
        "(a) Single-source configurations",
    )
    plot_level(
        ax_b,
        data["Pairwise"],
        "Pairwise",
        "(b) Pairwise configurations",
    )
    plot_level(
        ax_c,
        data["Triple"],
        "Triple",
        "(c) Triple-source configurations",
    )
    plot_level(
        ax_d,
        data["Quadruple"],
        "Quadruple",
        "(d) Quadruple-source configurations",
    )

    best_df = make_panel_e(data)
    plot_best_by_level(ax_e, best_df)

    # Give long source names enough room without wasting manuscript space.
    fig.subplots_adjust(
        left=0.145,
        right=0.985,
        top=0.965,
        bottom=0.075,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = args.output_dir / f"{args.prefix}.pdf"
    svg_path = args.output_dir / f"{args.prefix}.svg"
    png_path = args.output_dir / f"{args.prefix}.png"

    # PDF/SVG are vector outputs and are preferred for manuscript submission.
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    plt.close(fig)

    print("=" * 78)
    print("Unified 31-configuration figure created")
    print("=" * 78)
    print(f"PDF : {pdf_path}")
    print(f"SVG : {svg_path}")
    print(f"PNG : {png_path}")
    print("\nBest configuration by fusion level:")
    print(
        best_df[["Level", "Configuration", "Mean_AUC", "Std_AUC"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
