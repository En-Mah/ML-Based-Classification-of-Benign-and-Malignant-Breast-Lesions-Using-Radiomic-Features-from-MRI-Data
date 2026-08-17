#!/usr/bin/env python3
"""Build Figure 4: effect of SMOTE across 31 source configurations.

The script accepts either:
1) the batch ``*_full_best_per_fs_method.csv`` tables, or
2) the publication ``*_smote_vs_no_smote_summary.csv`` tables.

For every source configuration, it independently selects the highest
Mean_AUC under SMOTE and under no-SMOTE, then calculates:

    Delta Mean AUC = best SMOTE Mean_AUC - best no-SMOTE Mean_AUC

Run this script from the project root, or pass --project-root explicitly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVEL_ORDER = ["Single", "Pairwise", "Triple", "Quadruple", "All-five"]
EXPECTED_COUNTS = {
    "Single": 5,
    "Pairwise": 10,
    "Triple": 10,
    "Quadruple": 5,
    "All-five": 1,
}
EXPECTED_WINS = {
    "Single": {"SMOTE higher": 4, "No-SMOTE higher": 1},
    "Pairwise": {"SMOTE higher": 5, "No-SMOTE higher": 5},
    "Triple": {"SMOTE higher": 5, "No-SMOTE higher": 5},
    "Quadruple": {"SMOTE higher": 5, "No-SMOTE higher": 0},
    "All-five": {"SMOTE higher": 0, "No-SMOTE higher": 1},
}

# Candidate files are checked in this order. The first uniquely resolved file is used.
FILE_CANDIDATES = {
    "Single": [
        "single_dataset_full_best_per_fs_method.csv",
        "publication_single_dataset_smote_vs_no_smote_summary.csv",
        "single_dataset_smote_vs_no_smote_best_per_fs_method.csv",
    ],
    "Pairwise": [
        "all_10_pairwise_full_best_per_fs_method.csv",
        "publication_all_10_pairwise_smote_vs_no_smote_summary.csv",
        "all_10_pairwise_smote_vs_no_smote_best_per_fs_method.csv",
    ],
    "Triple": [
        "all_10_triple_full_best_per_fs_method.csv",
        "publication_all_10_triple_smote_vs_no_smote_summary.csv",
        "all_10_triple_smote_vs_no_smote_best_per_fs_method.csv",
    ],
    "Quadruple": [
        "all_5_quad_full_best_per_fs_method.csv",
        "publication_all_5_quad_smote_vs_no_smote_summary.csv",
        "all_5_quad_smote_vs_no_smote_best_per_fs_method.csv",
    ],
    "All-five": [
        "all_5_full_best_per_fs_method.csv",
        "publication_all_5_smote_vs_no_smote_summary.csv",
        "all_5_smote_vs_no_smote_best_per_fs_method.csv",
        "all_five_full_best_per_fs_method.csv",
        "publication_all_five_smote_vs_no_smote_summary.csv",
        "all_5_concat_full_best_per_fs_method.csv",
    ],
}

TAG_COLUMN_CANDIDATES = {
    "Single": ["Dataset", "Single_Tag", "Dataset_Tag", "Source_Configuration"],
    "Pairwise": ["Pair_Tag", "Pairwise_Tag", "Source_Configuration"],
    "Triple": ["Triple_Tag", "Triplewise_Tag", "Source_Configuration"],
    "Quadruple": ["Quad_Tag", "Quadruple_Tag", "Source_Configuration"],
    "All-five": [
        "All5_Tag",
        "All_Five_Tag",
        "AllFive_Tag",
        "Combination_Tag",
        "Source_Configuration",
    ],
}

SMOTE_COLOR = "#0072B2"       # colour-blind-safe blue
NO_SMOTE_COLOR = "#D55E00"    # colour-blind-safe vermillion
TIE_COLOR = "#7F7F7F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the publication figure for the SMOTE effect across 31 configurations."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing the batch-output folders. Default: current directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <project-root>/Figs",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-12,
        help="Absolute Delta AUC treated as a tie. Default: 1e-12.",
    )
    parser.add_argument(
        "--no-delta-labels",
        action="store_true",
        help="Do not print numeric Delta AUC values beside Panel (a) points.",
    )
    return parser.parse_args()


def _candidate_score(path: Path, level: str) -> tuple[int, float]:
    """Prefer final summary tables from SMOTE-comparison batch folders."""
    text = str(path).lower()
    score = 0
    if "summary_tables" in text or "_comparison_summary" in text:
        score += 5
    if "smote_compare" in text or "smote" in text:
        score += 3
    if level.lower().replace("-", "") in text.replace("-", ""):
        score += 2
    if ".ipynb_checkpoints" in text or "archive" in text or "backup" in text:
        score -= 20
    return score, path.stat().st_mtime


def resolve_input_file(project_root: Path, level: str) -> Path:
    """Find one suitable summary CSV for a fusion level."""
    for filename in FILE_CANDIDATES[level]:
        hits = [
            p.resolve()
            for p in project_root.rglob(filename)
            if p.is_file() and ".ipynb_checkpoints" not in str(p)
        ]
        if not hits:
            continue
        if len(hits) == 1:
            return hits[0]

        ranked = sorted(hits, key=lambda p: _candidate_score(p, level), reverse=True)
        top_score = _candidate_score(ranked[0], level)[0]
        top = [p for p in ranked if _candidate_score(p, level)[0] == top_score]
        if len(top) == 1:
            return top[0]

        message = "\n".join(f"  - {p}" for p in ranked)
        raise RuntimeError(
            f"Multiple equally suitable files were found for {level}:\n{message}\n"
            "Remove/rename old copies or set a unique final file in the project tree."
        )

    expected = "\n".join(f"  - {name}" for name in FILE_CANDIDATES[level])
    raise FileNotFoundError(
        f"Could not find an input summary for {level} under:\n  {project_root}\n"
        f"Expected one of:\n{expected}"
    )


def _normalise_smote_value(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if "no_smote" in text or text in {"false", "0", "no", "none"}:
        return False
    if text in {"true", "1", "yes", "smote", "with_smote"} or "_smote" in text:
        return True
    return None


def _join_dataset_columns(df: pd.DataFrame) -> pd.Series | None:
    dataset_cols = []
    for col in df.columns:
        name = str(col)
        if re.fullmatch(r"Dataset_[A-Z]", name, flags=re.IGNORECASE):
            dataset_cols.append(name)
        elif re.fullmatch(r"Source_[A-Z]", name, flags=re.IGNORECASE):
            dataset_cols.append(name)

    if not dataset_cols:
        return None

    dataset_cols = sorted(dataset_cols)
    return df[dataset_cols].astype(str).agg(" + ".join, axis=1)


def infer_configuration_series(df: pd.DataFrame, level: str) -> pd.Series:
    for col in TAG_COLUMN_CANDIDATES[level]:
        if col in df.columns and df[col].notna().any():
            return df[col].astype(str)

    joined = _join_dataset_columns(df)
    if joined is not None:
        return joined

    # The all-five level contains exactly one source configuration.
    if level == "All-five":
        return pd.Series(
            ["ADC + Pre + Post1 + Post2 + T2"] * len(df), index=df.index, dtype="object"
        )

    tag_like = [
        c for c in df.columns
        if str(c).endswith("_Tag")
        and str(c) not in {"Experiment_Tag", "Run_Tag"}
    ]
    if len(tag_like) == 1:
        return df[tag_like[0]].astype(str)

    raise KeyError(
        f"Could not infer the configuration label for {level}.\n"
        f"Available columns: {df.columns.tolist()}"
    )


def prettify_configuration_label(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"(?i)_M1\b", "", text)
    text = re.sub(
        r"(?i)_(single|pairwise|pair|triplewise|triple|quadruple|quad|allfive|all_?5)$",
        "",
        text,
    )
    text = text.replace("__", "_")
    text = re.sub(r"[_|]+", " + ", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s+", " ", text).strip(" +-_")
    return text


def _best_row(g: pd.DataFrame, metric_col: str) -> pd.Series:
    valid = g.dropna(subset=[metric_col]).copy()
    if valid.empty:
        raise ValueError("No valid Mean_AUC value is available for one balancing condition.")
    sort_cols = [c for c in [metric_col, "Pooled_AUC", "Mean_PR_AUC"] if c in valid.columns]
    return valid.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]


def aggregate_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Return one row per configuration with independently selected best SMOTE/no-SMOTE AUCs."""
    work = df.copy()
    work["Configuration_raw"] = infer_configuration_series(work, level)
    work["Configuration"] = work["Configuration_raw"].map(prettify_configuration_label)

    # Format A: publication delta table, one row per feature-selection strategy.
    direct_no = "Mean_AUC_No_SMOTE"
    direct_sm = "Mean_AUC_SMOTE"
    if {direct_no, direct_sm}.issubset(work.columns):
        work[direct_no] = pd.to_numeric(work[direct_no], errors="coerce")
        work[direct_sm] = pd.to_numeric(work[direct_sm], errors="coerce")

        rows = []
        for config, group in work.groupby("Configuration", sort=False, dropna=False):
            no_row = _best_row(group, direct_no)
            sm_row = _best_row(group, direct_sm)
            rows.append(
                {
                    "Fusion_Level": level,
                    "Configuration": config,
                    "Mean_AUC_No_SMOTE": float(no_row[direct_no]),
                    "Mean_AUC_SMOTE": float(sm_row[direct_sm]),
                    "Best_FS_No_SMOTE": no_row.get("FS_METHOD", np.nan),
                    "Best_FS_SMOTE": sm_row.get("FS_METHOD", np.nan),
                    "Best_Model_No_SMOTE": no_row.get("Best_Model_No_SMOTE", np.nan),
                    "Best_Model_SMOTE": sm_row.get("Best_Model_SMOTE", np.nan),
                }
            )
        out = pd.DataFrame(rows)

    # Format B: full best-per-feature-selection table with USE_SMOTE and Mean_AUC.
    elif {"USE_SMOTE", "Mean_AUC"}.issubset(work.columns):
        work["SMOTE_bool"] = work["USE_SMOTE"].map(_normalise_smote_value)
        work["Mean_AUC"] = pd.to_numeric(work["Mean_AUC"], errors="coerce")
        work = work.dropna(subset=["SMOTE_bool", "Mean_AUC"])

        rows = []
        for config, group in work.groupby("Configuration", sort=False, dropna=False):
            no_group = group[group["SMOTE_bool"] == False]  # noqa: E712
            sm_group = group[group["SMOTE_bool"] == True]   # noqa: E712
            if no_group.empty or sm_group.empty:
                raise ValueError(
                    f"Configuration '{config}' in {level} does not contain both SMOTE and no-SMOTE rows."
                )
            no_row = _best_row(no_group, "Mean_AUC")
            sm_row = _best_row(sm_group, "Mean_AUC")
            rows.append(
                {
                    "Fusion_Level": level,
                    "Configuration": config,
                    "Mean_AUC_No_SMOTE": float(no_row["Mean_AUC"]),
                    "Mean_AUC_SMOTE": float(sm_row["Mean_AUC"]),
                    "Best_FS_No_SMOTE": no_row.get("FS_METHOD", np.nan),
                    "Best_FS_SMOTE": sm_row.get("FS_METHOD", np.nan),
                    "Best_Model_No_SMOTE": no_row.get("Model", np.nan),
                    "Best_Model_SMOTE": sm_row.get("Model", np.nan),
                }
            )
        out = pd.DataFrame(rows)

    else:
        raise KeyError(
            f"Unsupported input schema for {level}. The file must contain either:\n"
            "  Mean_AUC_No_SMOTE and Mean_AUC_SMOTE, or\n"
            "  USE_SMOTE and Mean_AUC.\n"
            f"Available columns: {work.columns.tolist()}"
        )

    out["Delta_Mean_AUC"] = out["Mean_AUC_SMOTE"] - out["Mean_AUC_No_SMOTE"]
    return out


def classify_winner(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "SMOTE higher"
    if delta < -tolerance:
        return "No-SMOTE higher"
    return "Tie"


def validate_results(df: pd.DataFrame) -> None:
    errors = []
    if len(df) != 31:
        errors.append(f"Expected 31 configurations, but found {len(df)}.")

    actual_counts = df.groupby("Fusion_Level").size().to_dict()
    for level, expected in EXPECTED_COUNTS.items():
        actual = int(actual_counts.get(level, 0))
        if actual != expected:
            errors.append(f"{level}: expected {expected} configurations, found {actual}.")

    duplicated = df.duplicated(["Fusion_Level", "Configuration"], keep=False)
    if duplicated.any():
        duplicate_rows = df.loc[duplicated, ["Fusion_Level", "Configuration"]]
        errors.append("Duplicate configurations found:\n" + duplicate_rows.to_string(index=False))

    if errors:
        raise ValueError("\n".join(errors))


def print_win_check(df: pd.DataFrame) -> None:
    win_table = (
        df.groupby(["Fusion_Level", "Winner"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(LEVEL_ORDER, fill_value=0)
    )
    for col in ["SMOTE higher", "No-SMOTE higher", "Tie"]:
        if col not in win_table.columns:
            win_table[col] = 0
    win_table = win_table[["SMOTE higher", "No-SMOTE higher", "Tie"]]

    print("\nComputed win counts:")
    print(win_table.to_string())
    print("\nTotal:")
    print(win_table.sum(axis=0).to_string())

    mismatches = []
    for level in LEVEL_ORDER:
        for winner in ["SMOTE higher", "No-SMOTE higher"]:
            actual = int(win_table.loc[level, winner])
            expected = EXPECTED_WINS[level][winner]
            if actual != expected:
                mismatches.append(f"{level} / {winner}: expected {expected}, found {actual}")
    if int(win_table["Tie"].sum()) != 0:
        mismatches.append(f"Expected no ties, found {int(win_table['Tie'].sum())}.")

    if mismatches:
        print("\nWARNING: computed wins do not match the manuscript counts:", file=sys.stderr)
        for item in mismatches:
            print("  -", item, file=sys.stderr)
    else:
        print("\nWin-count check: matches 19 SMOTE wins and 12 no-SMOTE wins.")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_figure(df: pd.DataFrame, output_dir: Path, annotate_deltas: bool) -> None:
    configure_matplotlib()

    work = df.copy()
    work["Level_Order"] = work["Fusion_Level"].map({v: i for i, v in enumerate(LEVEL_ORDER)})
    work = work.sort_values(
        ["Level_Order", "Delta_Mean_AUC", "Configuration"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    work["y"] = np.arange(len(work))

    fig = plt.figure(figsize=(14, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.18)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # ------------------------------------------------------------------
    # Panel (a): configuration-level Delta Mean AUC dot/lollipop plot
    # ------------------------------------------------------------------
    max_abs = float(np.nanmax(np.abs(work["Delta_Mean_AUC"].to_numpy())))
    x_lim = max(max_abs * 1.28, 0.025)

    # Alternating bands and separators for the five fusion levels.
    for level_i, level in enumerate(LEVEL_ORDER):
        idx = work.index[work["Fusion_Level"] == level].to_numpy()
        if idx.size == 0:
            continue
        y0, y1 = idx.min() - 0.5, idx.max() + 0.5
        if level_i % 2 == 1:
            ax_a.axhspan(y0, y1, color="#F3F3F3", zorder=0)
        if level_i > 0:
            ax_a.axhline(y0, color="#B7B7B7", linewidth=0.7, zorder=1)
        ax_a.text(
            1.008,
            float(idx.mean()),
            level,
            transform=ax_a.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#555555",
            clip_on=False,
        )

    point_colours = work["Winner"].map(
        {
            "SMOTE higher": SMOTE_COLOR,
            "No-SMOTE higher": NO_SMOTE_COLOR,
            "Tie": TIE_COLOR,
        }
    )

    for _, row in work.iterrows():
        ax_a.hlines(
            row["y"],
            min(0.0, row["Delta_Mean_AUC"]),
            max(0.0, row["Delta_Mean_AUC"]),
            color=point_colours.loc[row.name],
            alpha=0.48,
            linewidth=1.5,
            zorder=2,
        )

    ax_a.scatter(
        work["Delta_Mean_AUC"],
        work["y"],
        s=38,
        c=point_colours,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax_a.axvline(0, color="#222222", linewidth=1.0, zorder=1)

    if annotate_deltas:
        offset = x_lim * 0.018
        for _, row in work.iterrows():
            delta = float(row["Delta_Mean_AUC"])
            ax_a.text(
                delta + (offset if delta >= 0 else -offset),
                row["y"],
                f"{delta:+.3f}",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=6.8,
                color="#333333",
            )

    ax_a.set_yticks(work["y"])
    ax_a.set_yticklabels(work["Configuration"])
    ax_a.invert_yaxis()
    ax_a.set_xlim(-x_lim, x_lim)
    ax_a.set_xlabel(r"$\Delta$ Mean outer-fold ROC-AUC (SMOTE - no-SMOTE)")
    ax_a.set_ylabel("Source configuration")
    ax_a.set_title("Configuration-level difference in mean outer-fold ROC-AUC", pad=18)
    ax_a.grid(axis="x", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax_a.set_axisbelow(True)
    ax_a.text(
        0.01,
        1.005,
        "No-SMOTE higher",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=NO_SMOTE_COLOR,
        fontweight="bold",
    )
    ax_a.text(
        0.99,
        1.005,
        "SMOTE higher",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=SMOTE_COLOR,
        fontweight="bold",
    )
    ax_a.text(
        -0.105,
        1.035,
        "(a)",
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Panel (b): grouped bars of wins by fusion level
    # ------------------------------------------------------------------
    wins = (
        work.groupby(["Fusion_Level", "Winner"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(LEVEL_ORDER, fill_value=0)
    )
    for col in ["SMOTE higher", "No-SMOTE higher"]:
        if col not in wins.columns:
            wins[col] = 0

    x = np.arange(len(LEVEL_ORDER))
    width = 0.36
    bars_sm = ax_b.bar(
        x - width / 2,
        wins["SMOTE higher"].to_numpy(),
        width,
        label="SMOTE higher",
        color=SMOTE_COLOR,
    )
    bars_no = ax_b.bar(
        x + width / 2,
        wins["No-SMOTE higher"].to_numpy(),
        width,
        label="No-SMOTE higher",
        color=NO_SMOTE_COLOR,
    )

    ax_b.bar_label(bars_sm, padding=3, fontsize=9)
    ax_b.bar_label(bars_no, padding=3, fontsize=9)

    totals = work.groupby("Fusion_Level").size().reindex(LEVEL_ORDER).fillna(0).astype(int)
    group_max = np.maximum(
        wins["SMOTE higher"].to_numpy(), wins["No-SMOTE higher"].to_numpy()
    )
    for xi, total, top in zip(x, totals.to_numpy(), group_max):
        ax_b.text(xi, top + 0.65, f"n={total}", ha="center", va="bottom", fontsize=8.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Single", "Pairwise", "Triple", "Quadruple", "All-five"], rotation=28, ha="right")
    ax_b.set_ylabel("Number of configurations")
    ax_b.set_title("Wins by feature-fusion level", pad=18)
    ax_b.set_ylim(0, max(6.8, float(group_max.max()) + 1.8))
    ax_b.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax_b.grid(axis="y", color="#D7D7D7", linewidth=0.55, alpha=0.8)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, loc="upper right", ncol=1)
    ax_b.text(
        -0.16,
        1.035,
        "(b)",
        transform=ax_b.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "smote_effect_across_31_configurations.pdf"
    png_path = output_dir / "smote_effect_across_31_configurations.png"
    svg_path = output_dir / "smote_effect_across_31_configurations.svg"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print("\nFigure saved:")
    print(" ", pdf_path)
    print(" ", png_path)
    print(" ", svg_path)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "Figs"
    )

    print("Project root:", project_root)
    print("\nInput files:")

    all_levels = []
    for level in LEVEL_ORDER:
        path = resolve_input_file(project_root, level)
        print(f"  {level:<11}: {path}")
        df = pd.read_csv(path)
        level_result = aggregate_level(df, level)
        all_levels.append(level_result)

    result = pd.concat(all_levels, ignore_index=True)
    result["Winner"] = result["Delta_Mean_AUC"].map(
        lambda x: classify_winner(float(x), args.tie_tolerance)
    )

    validate_results(result)
    print_win_check(result)

    source_csv = output_dir / "smote_effect_across_31_configurations_source_data.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(source_csv, index=False)
    print("\nPrepared source table saved:")
    print(" ", source_csv)

    draw_figure(result, output_dir, annotate_deltas=not args.no_delta_labels)


if __name__ == "__main__":
    main()
