#!/usr/bin/env python3
"""Build the classifier-selection frequency figure for 31 source configurations.

The script reads the five final publication-summary CSV files (single,
pairwise, triple, quadruple, and all-five), counts the classifier present in
each highest-ranked pipeline, and creates a publication-quality horizontal
stacked bar chart.

Primary output:
    Figs/classifier_win_counts_across_31_configurations.pdf

Additional outputs:
    Figs/classifier_win_counts_across_31_configurations.png
    Figs/classifier_win_counts_across_31_configurations.svg
    Figs/classifier_win_counts_audit.csv
    Figs/classifier_win_counts_table.csv

Run from the project root with automatic file discovery:
    python build_classifier_win_counts.py

Or provide the five files explicitly:
    python build_classifier_win_counts.py \
        --single path/to/publication_single_dataset_summary.csv \
        --pairwise path/to/publication_all_10_pairwise_concat_summary.csv \
        --triple path/to/publication_all_10_triple_concat_summary.csv \
        --quadruple path/to/publication_all_5_quad_concat_summary.csv \
        --all-five path/to/publication_all_5_concat_summary.csv \
        --output-dir Figs

Important:
    This script intentionally counts the rows already selected in the final
    publication summaries. It does not re-rank rows from full_all_models.csv.
    This preserves consistency with the manuscript tables and the existing
    publication-level figures.
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


# -----------------------------------------------------------------------------
# Study definition
# -----------------------------------------------------------------------------
FUSION_LEVELS = ["Single", "Pairwise", "Triple", "Quadruple", "All-five"]
EXPECTED_ROWS = {
    "Single": 5,
    "Pairwise": 10,
    "Triple": 10,
    "Quadruple": 5,
    "All-five": 1,
}

CLASSIFIER_ORDER = [
    "RBF SVM",
    "Logistic Regression",
    "Linear SVM",
    "Gaussian Naive Bayes",
    "KNN",
    "XGBoost",
    "LightGBM",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
]

# Fixed, color-blind-friendly colors used consistently across the manuscript.
FUSION_COLORS = {
    "Single": "#4C78A8",
    "Pairwise": "#F58518",
    "Triple": "#54A24B",
    "Quadruple": "#E45756",
    "All-five": "#B279A2",
}

# Reference counts currently reported in the manuscript draft. They are used
# only when --check-reference-counts is supplied; they are not used to build
# the figure.
REFERENCE_COUNTS = pd.DataFrame(
    {
        "Single": [2, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        "Pairwise": [4, 2, 1, 1, 1, 1, 0, 0, 0, 0],
        "Triple": [1, 3, 3, 2, 0, 0, 1, 0, 0, 0],
        "Quadruple": [3, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "All-five": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    index=CLASSIFIER_ORDER,
)

MODEL_COLUMN_CANDIDATES = [
    "Model",
    "Classifier",
    "Best_Model",
    "Best_Classifier",
    "model",
    "classifier",
]

CONFIG_COLUMN_CANDIDATES = [
    "Source_Configuration",
    "Configuration",
    "Dataset",
    "Dataset_Tag",
    "Single_Tag",
    "Pair_Tag",
    "Triple_Tag",
    "Quad_Tag",
    "Quadruple_Tag",
    "Combination",
    "Concat_Tag",
]

AUTO_DISCOVERY_NAMES = {
    "Single": [
        "publication_single_dataset_summary.csv",
        "publication_all_5_single_dataset_summary.csv",
    ],
    "Pairwise": [
        "publication_all_10_pairwise_concat_summary.csv",
        "publication_pairwise_concat_summary.csv",
    ],
    "Triple": [
        "publication_all_10_triple_concat_summary.csv",
        "publication_triple_concat_summary.csv",
    ],
    "Quadruple": [
        "publication_all_5_quad_concat_summary.csv",
        "publication_all_5_quadruple_concat_summary.csv",
    ],
    "All-five": [
        "publication_all_5_concat_summary.csv",
        "publication_all_five_concat_summary.csv",
    ],
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalized_token(value: object) -> str:
    """Return a lowercase alphanumeric token for robust name matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonical_classifier_name(raw_name: object) -> str:
    """Map notebook/model aliases to the ten manuscript classifier names."""
    if pd.isna(raw_name):
        raise ValueError("Classifier/model value is missing.")

    token = normalized_token(raw_name)

    aliases = {
        "RBF SVM": {
            "rbfsvm", "svmrbf", "svc", "rbfsvc", "svcrbf",
        },
        "Linear SVM": {
            "linearsvm", "svmlinear", "linearsvc", "svmlinearclassifier",
        },
        "Logistic Regression": {
            "logisticregression", "logistic", "logreg", "lr",
        },
        "Gaussian Naive Bayes": {
            "gaussiannaivebayes", "gaussiannb", "naivebayes", "gnb",
        },
        "KNN": {
            "knn", "kneighbors", "kneighborsclassifier",
            "knearestneighbours", "knearestneighbors",
        },
        "XGBoost": {
            "xgboost", "xgb", "xgbclassifier",
        },
        "LightGBM": {
            "lightgbm", "lgbm", "lgbmclassifier",
        },
        "Random Forest": {
            "randomforest", "randomforestclassifier", "rf",
        },
        "Extra Trees": {
            "extratrees", "extratreesclassifier", "et",
        },
        "Gradient Boosting": {
            "gradientboosting", "gradientboostingclassifier", "gb",
            "gbclassifier",
        },
    }

    for canonical, tokens in aliases.items():
        if token in tokens:
            return canonical

    # Conservative pattern-based fallbacks for common exported names.
    if "svm" in token or "svc" in token:
        if "rbf" in token or token == "svc":
            return "RBF SVM"
        if "linear" in token:
            return "Linear SVM"
    if "logistic" in token:
        return "Logistic Regression"
    if "naivebayes" in token or "gaussiannb" in token:
        return "Gaussian Naive Bayes"
    if "neighbor" in token or token.startswith("knn"):
        return "KNN"
    if "xgb" in token:
        return "XGBoost"
    if "lightgbm" in token or "lgbm" in token:
        return "LightGBM"
    if "randomforest" in token:
        return "Random Forest"
    if "extratrees" in token:
        return "Extra Trees"
    if "gradientboost" in token:
        return "Gradient Boosting"

    raise ValueError(
        f"Unrecognized classifier name: {raw_name!r}. "
        "Add its alias to canonical_classifier_name()."
    )


def detect_column(columns: Iterable[str], candidates: list[str], purpose: str) -> str:
    """Detect a column using exact and normalized matching."""
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate

    normalized_columns = {normalized_token(col): col for col in columns}
    for candidate in candidates:
        key = normalized_token(candidate)
        if key in normalized_columns:
            return normalized_columns[key]

    raise KeyError(
        f"Could not detect the {purpose} column.\n"
        f"Available columns: {columns}\n"
        f"Expected one of: {candidates}"
    )


def detect_optional_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    try:
        return detect_column(columns, candidates, "configuration")
    except KeyError:
        return None


def discover_summary_file(project_root: Path, fusion_level: str) -> Path:
    """Find one publication-summary file by its expected filename."""
    matches: list[Path] = []
    for filename in AUTO_DISCOVERY_NAMES[fusion_level]:
        matches.extend(project_root.rglob(filename))

    # Remove duplicate paths while retaining deterministic order.
    matches = sorted({p.resolve() for p in matches if p.is_file()})

    if not matches:
        expected = " or ".join(AUTO_DISCOVERY_NAMES[fusion_level])
        raise FileNotFoundError(
            f"Could not automatically find the {fusion_level} publication summary "
            f"under {project_root.resolve()}. Expected filename: {expected}. "
            f"Pass the path explicitly with the corresponding command-line option."
        )

    if len(matches) > 1:
        formatted = "\n".join(f"  - {p}" for p in matches)
        raise RuntimeError(
            f"Multiple candidate files were found for {fusion_level}:\n{formatted}\n"
            "Pass the intended file explicitly to avoid using an old summary."
        )

    return matches[0]


def read_publication_summary(path: Path, fusion_level: str) -> pd.DataFrame:
    """Read and validate one final publication-level summary CSV."""
    if not path.exists():
        raise FileNotFoundError(f"{fusion_level} summary not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{fusion_level} summary is empty: {path}")

    model_col = detect_column(df.columns, MODEL_COLUMN_CANDIDATES, "classifier/model")
    config_col = detect_optional_column(df.columns, CONFIG_COLUMN_CANDIDATES)

    # Failed rows or metadata-only rows cannot contribute to classifier counts.
    usable = df.loc[df[model_col].notna()].copy()
    usable = usable.loc[usable[model_col].astype(str).str.strip().ne("")].copy()

    expected_n = EXPECTED_ROWS[fusion_level]
    if len(usable) != expected_n:
        raise ValueError(
            f"{fusion_level}: expected exactly {expected_n} selected publication rows, "
            f"but found {len(usable)} usable rows in:\n{path}\n"
            "This script does not silently re-rank full-model tables. Confirm that you "
            "provided the final one-row-per-source-configuration publication summary."
        )

    if config_col is not None:
        configuration = usable[config_col].astype(str).str.strip()
        duplicate_mask = configuration.duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_values = sorted(configuration.loc[duplicate_mask].unique())
            raise ValueError(
                f"{fusion_level}: duplicated source configurations were found in "
                f"column {config_col!r}: {duplicate_values}"
            )
    else:
        # A single all-five row may reasonably lack a configuration tag.
        if expected_n == 1:
            configuration = pd.Series(["All five sources"], index=usable.index)
        else:
            configuration = pd.Series(
                [f"{fusion_level} configuration {i + 1}" for i in range(len(usable))],
                index=usable.index,
            )

    canonical_models = usable[model_col].map(canonical_classifier_name)

    audit = pd.DataFrame(
        {
            "Fusion_Level": fusion_level,
            "Configuration": configuration.to_numpy(),
            "Raw_Model": usable[model_col].astype(str).to_numpy(),
            "Classifier": canonical_models.to_numpy(),
            "Source_File": str(path.resolve()),
            "Source_Row": usable.index.to_numpy() + 2,  # +2 for CSV header and 0-based index
        }
    )

    return audit


def build_count_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Create classifier-by-fusion-level counts in manuscript order."""
    counts = pd.crosstab(audit["Classifier"], audit["Fusion_Level"])
    counts = counts.reindex(index=CLASSIFIER_ORDER, columns=FUSION_LEVELS, fill_value=0)
    counts = counts.astype(int)
    counts["Total"] = counts.sum(axis=1)

    total_configurations = int(counts["Total"].sum())
    if total_configurations != 31:
        raise AssertionError(
            f"Expected 31 configurations in total, obtained {total_configurations}."
        )

    level_totals = counts[FUSION_LEVELS].sum(axis=0).to_dict()
    for level, expected in EXPECTED_ROWS.items():
        observed = int(level_totals[level])
        if observed != expected:
            raise AssertionError(
                f"{level}: expected {expected} configurations, obtained {observed}."
            )

    return counts


def check_reference_counts(counts: pd.DataFrame) -> None:
    """Compare data-derived counts with the currently reported manuscript counts."""
    observed = counts[FUSION_LEVELS]
    difference = observed - REFERENCE_COUNTS

    if (difference.to_numpy() == 0).all():
        print("Reference-count check: PASS — counts match the current manuscript draft.")
        return

    changed = difference.loc[(difference != 0).any(axis=1)]
    print("\nReference-count check: DIFFERENCES FOUND")
    print("Positive values mean the current data contain more selections than the draft.")
    print("Negative values mean the current data contain fewer selections than the draft.\n")
    print(changed.to_string())


def configure_matplotlib() -> None:
    """Set journal-friendly, portable rendering defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
        }
    )


def plot_classifier_counts(counts: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Create PDF, PNG, and SVG versions of the stacked horizontal bar chart."""
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_counts = counts.loc[CLASSIFIER_ORDER, FUSION_LEVELS]
    totals = counts.loc[CLASSIFIER_ORDER, "Total"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=False)
    y = np.arange(len(CLASSIFIER_ORDER))
    left = np.zeros(len(CLASSIFIER_ORDER), dtype=float)

    for fusion_level in FUSION_LEVELS:
        values = plot_counts[fusion_level].to_numpy(dtype=float)
        bars = ax.barh(
            y,
            values,
            left=left,
            height=0.64,
            color=FUSION_COLORS[fusion_level],
            edgecolor="white",
            linewidth=0.8,
            label=fusion_level,
            zorder=3,
        )

        for row_index, (bar, value) in enumerate(zip(bars, values)):
            if value <= 0:
                continue
            ax.text(
                left[row_index] + value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
                zorder=4,
            )

        left += values

    for row_index, total in enumerate(totals):
        x_position = 0.18 if total == 0 else total + 0.18
        ax.text(
            x_position,
            row_index,
            f"{total}",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(CLASSIFIER_ORDER)
    ax.invert_yaxis()

    max_total = int(totals.max())
    x_max = max(12.4, max_total + 1.4)
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, int(np.floor(x_max)) + 1, 2))

    ax.set_xlabel("Number of configurations ranked first")
    ax.set_title(
        "Highest-ranked classifier across 31 source configurations",
        fontweight="bold",
        pad=15,
    )

    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
    ax.yaxis.grid(False)
    ax.tick_params(axis="y", pad=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        title="Feature-fusion level",
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        edgecolor="0.25",
        fancybox=False,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.20, right=0.975, top=0.91, bottom=0.12)

    stem = output_dir / "classifier_win_counts_across_31_configurations"
    paths = {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
    }

    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the classifier win-count figure from the five final "
            "publication-summary CSV files."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used for automatic recursive file discovery.",
    )
    parser.add_argument("--single", type=Path, help="Single-source publication summary CSV.")
    parser.add_argument("--pairwise", type=Path, help="Pairwise publication summary CSV.")
    parser.add_argument("--triple", type=Path, help="Triple-source publication summary CSV.")
    parser.add_argument("--quadruple", type=Path, help="Quadruple-source publication summary CSV.")
    parser.add_argument(
        "--all-five",
        dest="all_five",
        type=Path,
        help="Complete five-source publication summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Figs"),
        help="Output directory. Default: Figs",
    )
    parser.add_argument(
        "--check-reference-counts",
        action="store_true",
        help="Compare data-derived counts with the current manuscript count table.",
    )
    return parser.parse_args()


def resolve_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    explicit = {
        "Single": args.single,
        "Pairwise": args.pairwise,
        "Triple": args.triple,
        "Quadruple": args.quadruple,
        "All-five": args.all_five,
    }

    resolved: dict[str, Path] = {}
    for fusion_level in FUSION_LEVELS:
        path = explicit[fusion_level]
        if path is None:
            path = discover_summary_file(args.project_root, fusion_level)
        resolved[fusion_level] = path.resolve()

    return resolved


def main() -> int:
    args = parse_args()

    try:
        input_paths = resolve_input_paths(args)

        print("Input publication summaries:")
        for fusion_level in FUSION_LEVELS:
            print(f"  {fusion_level:<11}: {input_paths[fusion_level]}")

        audit_parts = [
            read_publication_summary(input_paths[level], level)
            for level in FUSION_LEVELS
        ]
        audit = pd.concat(audit_parts, ignore_index=True)
        counts = build_count_table(audit)

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        audit_path = output_dir / "classifier_win_counts_audit.csv"
        table_path = output_dir / "classifier_win_counts_table.csv"
        audit.to_csv(audit_path, index=False)
        counts.to_csv(table_path, index_label="Classifier")

        figure_paths = plot_classifier_counts(counts, output_dir)

        print("\nData-derived classifier counts:")
        print(counts.to_string())

        if args.check_reference_counts:
            check_reference_counts(counts)

        print("\nSaved outputs:")
        print(f"  PDF  : {figure_paths['pdf']}")
        print(f"  PNG  : {figure_paths['png']}")
        print(f"  SVG  : {figure_paths['svg']}")
        print(f"  Audit: {audit_path}")
        print(f"  Table: {table_path}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
