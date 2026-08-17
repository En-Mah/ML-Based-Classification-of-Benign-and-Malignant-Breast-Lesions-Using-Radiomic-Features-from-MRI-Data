"""Build the manuscript figure comparing feature-selection strategies.

Expected input: five batch-level ``*_full_best_per_fs_method.csv`` files.
Each input row must already be the best classifier for one
(source configuration, feature-selection method, SMOTE condition) combination.

The script:
1. combines the five fusion levels (expected total: 186 rows),
2. retains the better SMOTE condition for each method/configuration (93 rows),
3. identifies the winning method for each of the 31 configurations,
4. validates the manuscript statistics and win counts,
5. saves a vector PDF, a 600-dpi PNG, and audit/supplementary CSV files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================
# Run the script from the repository root, or replace Path.cwd() with the
# absolute repository path.
PROJECT_ROOT = Path.cwd()

# Optional: set any value below to an exact path. Leave as None to search the
# repository recursively by the candidate filenames listed further below.
FILE_OVERRIDES: dict[str, Path | None] = {
    "Single": None,
    "Pairwise": None,
    "Triple": None,
    "Quadruple": None,
    "All-five": None,
}

# Keep True for the final manuscript figure. The script will stop if the CSVs
# do not reproduce the numbers reported in the Results subsection.
STRICT_EXPECTED_RESULTS = True

FIGURE_DIR = PROJECT_ROOT / "figures"
AUDIT_DIR = PROJECT_ROOT / "output" / "manuscript_summary_tables"

FIGURE_PDF = FIGURE_DIR / "feature_selection_strategy_comparison.pdf"
FIGURE_PNG = FIGURE_DIR / "feature_selection_strategy_comparison.png"


# =============================================================================
# INPUT FILE DISCOVERY
# =============================================================================
FILENAME_CANDIDATES: dict[str, list[str]] = {
    "Single": [
        "single_dataset_full_best_per_fs_method.csv",
    ],
    "Pairwise": [
        "all_10_pairwise_full_best_per_fs_method.csv",
    ],
    "Triple": [
        "all_10_triple_full_best_per_fs_method.csv",
    ],
    "Quadruple": [
        "all_5_quad_full_best_per_fs_method.csv",
        "all_5_quadruple_full_best_per_fs_method.csv",
    ],
    "All-five": [
        "all_5_full_best_per_fs_method.csv",
        "all_5_concat_full_best_per_fs_method.csv",
        "all_five_full_best_per_fs_method.csv",
    ],
}

EXPECTED_CONFIG_COUNTS = {
    "Single": 5,
    "Pairwise": 10,
    "Triple": 10,
    "Quadruple": 5,
    "All-five": 1,
}

FUSION_ORDER = ["Single", "Pairwise", "Triple", "Quadruple", "All-five"]
METHOD_ORDER_PANEL_A = ["Corr. → mRMR", "EN → mRMR", "mRMR → EN"]
METHOD_ORDER_PANEL_B = ["EN → mRMR", "mRMR → EN", "Corr. → mRMR"]


# =============================================================================
# HELPERS
# =============================================================================
def resolve_input_file(level: str) -> Path:
    """Resolve one unique CSV for a fusion level."""
    override = FILE_OVERRIDES[level]
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Override does not exist for {level}: {path}")
        return path

    matches: list[Path] = []
    for filename in FILENAME_CANDIDATES[level]:
        matches.extend(PROJECT_ROOT.rglob(filename))

    matches = sorted({p.resolve() for p in matches})

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find the {level} full-best CSV below {PROJECT_ROOT}.\n"
            f"Expected one of: {FILENAME_CANDIDATES[level]}\n"
            "Set FILE_OVERRIDES to the exact path if the filename is different."
        )

    # Prefer a unique file located in a batch runner's summary_tables folder.
    summary_matches = [p for p in matches if "summary_tables" in p.parts]
    if len(summary_matches) == 1:
        return summary_matches[0]

    formatted = "\n".join(f"  - {p}" for p in matches)
    raise RuntimeError(
        f"Multiple candidate files were found for {level}:\n{formatted}\n"
        "Set FILE_OVERRIDES to the exact final file."
    )


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return str(lookup[key])
    return None


def resolve_configuration_column(df: pd.DataFrame, level: str) -> str | None:
    candidates = {
        "Single": ["Single_Tag", "Dataset", "Dataset_Tag", "Source_Configuration"],
        "Pairwise": ["Pair_Tag", "Pair", "Dataset_Tag", "Source_Configuration"],
        "Triple": ["Triple_Tag", "Triple", "Dataset_Tag", "Source_Configuration"],
        "Quadruple": [
            "Quad_Tag",
            "Quadruple_Tag",
            "Quadruple",
            "Dataset_Tag",
            "Source_Configuration",
        ],
        "All-five": [
            "All5_Tag",
            "All_Five_Tag",
            "AllFive_Tag",
            "Combination",
            "Dataset_Tag",
            "Source_Configuration",
        ],
    }[level]

    direct = first_existing_column(df, candidates)
    if direct is not None:
        return direct

    # Fallback: a tag-like column with exactly the expected number of unique
    # configurations for this fusion level.
    expected_n = EXPECTED_CONFIG_COUNTS[level]
    tag_like = [
        c
        for c in df.columns
        if str(c).lower().endswith("_tag")
        and df[c].nunique(dropna=True) == expected_n
    ]
    if len(tag_like) == 1:
        return str(tag_like[0])

    return None


def clean_configuration_label(value: object, level: str) -> str:
    if level == "All-five" and (pd.isna(value) or str(value).strip() == ""):
        return "ADC–Post1–Post2–Pre–T2"

    text = str(value).strip()
    for suffix in [
        "_M1_pairwise",
        "_M1_triple",
        "_M1_quad",
        "_pairwise",
        "_triple",
        "_quadruple",
        "_quad",
        "_all_five",
        "_allfive",
        "_all_5",
        "_all5",
        "_M1",
    ]:
        if text.lower().endswith(suffix.lower()):
            text = text[: -len(suffix)]
    text = text.replace("__", "_").strip("_")
    return text.replace("_", "–")


def normalize_fs_method(value: object) -> str:
    """Map the raw experiment name to one of the three manuscript labels.

    Important: names such as ``mrmr_then_elasticnet_no_corr_auto_k`` contain
    the substring ``corr`` inside ``no_corr``. Therefore, the method order
    must be detected before checking for a genuine correlation-filter step.
    """
    import re

    raw = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    mrmr_match = re.search(r"(?:^|_)mrmr(?:_|$)", text)
    en_match = re.search(r"(?:^|_)elastic_?net(?:_|$)", text)

    # Detect the two ordered Elastic Net/mRMR pipelines first. This prevents
    # ``no_corr`` from being misread as the correlation-filter strategy.
    if mrmr_match is not None and en_match is not None:
        if en_match.start() < mrmr_match.start():
            return "EN → mRMR"
        return "mRMR → EN"

    # Remove negative flags before looking for a genuine correlation step.
    corr_text = re.sub(
        r"(?:^|_)no_?(?:corr|correlation)(?=_|$)",
        "_",
        text,
    )
    corr_match = re.search(
        r"(?:^|_)(?:corr|correlation)(?:elation)?(?:_|$)",
        corr_text,
    )

    if corr_match is not None and mrmr_match is not None:
        return "Corr. → mRMR"

    raise ValueError(
        "Unrecognised FS_METHOD value: "
        f"{value!r}. Expected an ordered Elastic Net/mRMR name or a "
        "correlation-filter/mRMR name."
    )


def normalize_smote(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError("USE_SMOTE contains a missing value.")

    text = str(value).strip().lower()
    if text in {"false", "0", "no", "n", "no_smote", "without_smote"}:
        return False
    if text in {"true", "1", "yes", "y", "smote", "with_smote"}:
        return True
    raise ValueError(f"Unrecognised USE_SMOTE value: {value!r}")


def load_and_standardize(level: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    required = ["FS_METHOD", "USE_SMOTE", "Model", "Mean_AUC"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    config_col = resolve_configuration_column(df, level)
    if config_col is None:
        if level == "All-five":
            df["Source_Configuration"] = "ADC–Post1–Post2–Pre–T2"
        else:
            raise ValueError(
                f"Could not identify the source-configuration column in {path}.\n"
                f"Available columns: {list(df.columns)}"
            )
    else:
        df["Source_Configuration"] = df[config_col].map(
            lambda x: clean_configuration_label(x, level)
        )

    df["Fusion_Level"] = level
    df["Method"] = df["FS_METHOD"].map(normalize_fs_method)
    df["USE_SMOTE_BOOL"] = df["USE_SMOTE"].map(normalize_smote)
    df["Selected_SMOTE_Condition"] = np.where(
        df["USE_SMOTE_BOOL"], "SMOTE", "No SMOTE"
    )

    numeric_candidates = [
        "Mean_AUC",
        "Pooled_AUC",
        "Mean_PR_AUC",
        "Pooled_PR_AUC",
        "Std_AUC",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Mean_AUC"].isna().any():
        bad_n = int(df["Mean_AUC"].isna().sum())
        raise ValueError(f"{bad_n} non-numeric/missing Mean_AUC values in {path}")

    observed_configs = df["Source_Configuration"].nunique()
    expected_configs = EXPECTED_CONFIG_COUNTS[level]
    if observed_configs != expected_configs:
        raise ValueError(
            f"{level}: expected {expected_configs} source configurations, "
            f"but found {observed_configs} in {path}."
        )

    return df


def sorted_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["Mean_AUC"]
    for col in ["Pooled_AUC", "Mean_PR_AUC", "Pooled_PR_AUC"]:
        if col in df.columns:
            sort_cols.append(col)
    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")


# =============================================================================
# BUILD THE 186-ROW AND 93-ROW TABLES
# =============================================================================
input_paths = {level: resolve_input_file(level) for level in FUSION_ORDER}

print("Input files:")
for level, path in input_paths.items():
    print(f"  {level:11s}: {path}")

raw_parts = [load_and_standardize(level, input_paths[level]) for level in FUSION_ORDER]
raw_186 = pd.concat(raw_parts, ignore_index=True)
raw_186["Fusion_Level"] = pd.Categorical(
    raw_186["Fusion_Level"], categories=FUSION_ORDER, ordered=True
)

print("\nFeature-selection name mapping:")
method_mapping = (
    raw_186[["FS_METHOD", "Method"]]
    .drop_duplicates()
    .sort_values(["Method", "FS_METHOD"])
)
print(method_mapping.to_string(index=False))

print("\nRows loaded by fusion level:")
print(raw_186.groupby("Fusion_Level", observed=False).size())

# The batch-level full-best files should contain:
# 31 configurations × 3 methods × 2 SMOTE conditions = 186 rows.
if len(raw_186) != 186:
    raise ValueError(f"Expected 186 input rows, but found {len(raw_186)}.")

raw_key = [
    "Fusion_Level",
    "Source_Configuration",
    "Method",
    "USE_SMOTE_BOOL",
]
duplicates = raw_186.duplicated(raw_key, keep=False)
if duplicates.any():
    duplicate_rows = raw_186.loc[
        duplicates,
        raw_key + ["Model", "Mean_AUC"],
    ].sort_values(raw_key)
    raise ValueError(
        "Duplicate method–SMOTE rows were found. Each key must occur once.\n"
        + duplicate_rows.to_string(index=False)
    )

method_key = ["Fusion_Level", "Source_Configuration", "Method"]
method_specific_93 = (
    sorted_for_selection(raw_186)
    .drop_duplicates(method_key, keep="first")
    .sort_values(method_key)
    .reset_index(drop=True)
)

if len(method_specific_93) != 93:
    raise ValueError(
        f"Expected 93 method-specific rows after SMOTE selection, "
        f"but found {len(method_specific_93)}."
    )

# Fusion_Level is categorical. With observed=False, pandas can generate
# unobserved Fusion_Level × Source_Configuration combinations and assign
# them a method count of zero. Use observed=True so only real configurations
# present in the data are validated.
per_config_method_count = method_specific_93.groupby(
    ["Fusion_Level", "Source_Configuration"], observed=True
)["Method"].nunique()

bad_method_counts = per_config_method_count[per_config_method_count.ne(3)]
if not bad_method_counts.empty:
    bad_keys = bad_method_counts.reset_index()[
        ["Fusion_Level", "Source_Configuration"]
    ]
    bad_rows = method_specific_93.merge(
        bad_keys,
        on=["Fusion_Level", "Source_Configuration"],
        how="inner",
    )[
        [
            "Fusion_Level",
            "Source_Configuration",
            "Method",
            "FS_METHOD",
            "USE_SMOTE_BOOL",
            "Model",
            "Mean_AUC",
        ]
    ].sort_values(["Fusion_Level", "Source_Configuration", "Method"])

    raise ValueError(
        "At least one observed source configuration does not contain all "
        "three methods.\nMethod counts:\n"
        + bad_method_counts.to_string()
        + "\n\nAvailable rows for the affected configurations:\n"
        + bad_rows.to_string(index=False)
    )

# Detect exact top-Mean-AUC ties before selecting one winning method.
def top_tie_count(series: pd.Series) -> int:
    return int(np.isclose(series, series.max(), rtol=0.0, atol=1e-12).sum())

winner_group_key = ["Fusion_Level", "Source_Configuration"]
tie_counts = method_specific_93.groupby(
    winner_group_key, observed=True
)["Mean_AUC"].apply(top_tie_count)
if (tie_counts > 1).any():
    tied_configs = tie_counts[tie_counts > 1]
    raise ValueError(
        "Exact ties in the highest Mean_AUC were found. Resolve/report ties before "
        "creating a single-winner plot:\n"
        + tied_configs.to_string()
    )

winners_31 = (
    sorted_for_selection(method_specific_93)
    .drop_duplicates(winner_group_key, keep="first")
    .sort_values(winner_group_key)
    .reset_index(drop=True)
)

if len(winners_31) != 31:
    raise ValueError(f"Expected 31 winning rows, but found {len(winners_31)}.")


# =============================================================================
# SUMMARIES AND MANUSCRIPT VALIDATION
# =============================================================================
summary_stats = (
    method_specific_93.groupby("Method")["Mean_AUC"]
    .agg(N="count", Mean="mean", Median="median", Maximum="max")
    .reindex(METHOD_ORDER_PANEL_A)
)

win_counts = (
    pd.crosstab(winners_31["Method"], winners_31["Fusion_Level"])
    .reindex(index=METHOD_ORDER_PANEL_B, columns=FUSION_ORDER, fill_value=0)
    .astype(int)
)
win_counts["Total"] = win_counts.sum(axis=1)

print("\nMethod-specific summary statistics:")
print(summary_stats.round(3))
print("\nWin counts by fusion level:")
print(win_counts)

if STRICT_EXPECTED_RESULTS:
    expected_stats = {
        "Corr. → mRMR": {"Mean": 0.694, "Median": 0.698, "Maximum": 0.787},
        "EN → mRMR": {"Mean": 0.754, "Median": 0.760, "Maximum": 0.836},
        "mRMR → EN": {"Mean": 0.736, "Median": 0.747, "Maximum": 0.793},
    }
    for method, expected in expected_stats.items():
        for metric, expected_value in expected.items():
            actual_value = round(float(summary_stats.loc[method, metric]), 3)
            if actual_value != expected_value:
                raise AssertionError(
                    f"Manuscript mismatch for {method}, {metric}: "
                    f"expected {expected_value:.3f}, obtained {actual_value:.3f}."
                )

    expected_wins = pd.DataFrame(
        {
            "Single": [3, 2, 0],
            "Pairwise": [6, 4, 0],
            "Triple": [6, 4, 0],
            "Quadruple": [4, 1, 0],
            "All-five": [1, 0, 0],
            "Total": [20, 11, 0],
        },
        index=METHOD_ORDER_PANEL_B,
    )
    if not win_counts.equals(expected_wins):
        raise AssertionError(
            "Win-count table does not match the Results subsection.\n\n"
            f"Expected:\n{expected_wins}\n\nObtained:\n{win_counts}"
        )


# =============================================================================
# SAVE AUDIT AND SUPPLEMENTARY TABLES
# =============================================================================
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

raw_186.to_csv(AUDIT_DIR / "feature_selection_input_186_rows.csv", index=False)
method_specific_93.to_csv(
    AUDIT_DIR / "feature_selection_method_specific_93_rows.csv", index=False
)
summary_stats.reset_index().to_csv(
    AUDIT_DIR / "feature_selection_summary_statistics.csv", index=False
)
win_counts.reset_index().rename(columns={"index": "Method"}).to_csv(
    AUDIT_DIR / "feature_selection_win_counts_by_fusion_level.csv", index=False
)

# Supplementary Table Sx: one row per source configuration.
auc_wide = (
    method_specific_93.pivot(
        index=["Fusion_Level", "Source_Configuration"],
        columns="Method",
        values="Mean_AUC",
    )
    .reset_index()
)

winner_details = winners_31[
    [
        "Fusion_Level",
        "Source_Configuration",
        "Method",
        "Selected_SMOTE_Condition",
        "Model",
    ]
].rename(
    columns={
        "Method": "Winning_Strategy",
        "Selected_SMOTE_Condition": "Selected_SMOTE_Condition",
        "Model": "Winning_Classifier",
    }
)

supplementary_table = auc_wide.merge(
    winner_details,
    on=["Fusion_Level", "Source_Configuration"],
    how="left",
)
supplementary_table = supplementary_table.rename(
    columns={
        "Corr. → mRMR": "Corr_to_mRMR_best_Mean_AUC",
        "EN → mRMR": "EN_to_mRMR_best_Mean_AUC",
        "mRMR → EN": "mRMR_to_EN_best_Mean_AUC",
    }
)
supplementary_table["Fusion_Level"] = pd.Categorical(
    supplementary_table["Fusion_Level"], categories=FUSION_ORDER, ordered=True
)
supplementary_table = supplementary_table.sort_values(
    ["Fusion_Level", "Source_Configuration"]
)
supplementary_table.to_csv(
    AUDIT_DIR / "supplementary_feature_selection_performance_all_31_configurations.csv",
    index=False,
)


# =============================================================================
# FIGURE: PANEL (a) DISTRIBUTIONS; PANEL (b) STACKED WIN COUNTS
# =============================================================================
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

method_colors = {
    "Corr. → mRMR": "#8B8B8B",
    "EN → mRMR": "#2A9D8F",
    "mRMR → EN": "#457B9D",
}
fusion_colors = {
    "Single": "#4C78A8",
    "Pairwise": "#F58518",
    "Triple": "#54A24B",
    "Quadruple": "#E45756",
    "All-five": "#B279A2",
}

fig, (ax_a, ax_b) = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.6),
    gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.34},
)

# Panel (a)
box_data = [
    method_specific_93.loc[
        method_specific_93["Method"] == method, "Mean_AUC"
    ].to_numpy()
    for method in METHOD_ORDER_PANEL_A
]
positions = np.arange(1, len(METHOD_ORDER_PANEL_A) + 1)

boxplot = ax_a.boxplot(
    box_data,
    positions=positions,
    widths=0.56,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "#1F1F1F", "linewidth": 1.5},
    boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
    whiskerprops={"linewidth": 1.0, "color": "#555555"},
    capprops={"linewidth": 1.0, "color": "#555555"},
)
for patch, method in zip(boxplot["boxes"], METHOD_ORDER_PANEL_A):
    patch.set_facecolor(method_colors[method])
    patch.set_alpha(0.42)

rng = np.random.default_rng(20260806)
for pos, method, values in zip(positions, METHOD_ORDER_PANEL_A, box_data):
    jitter = rng.uniform(-0.17, 0.17, size=len(values))
    ax_a.scatter(
        np.full(len(values), pos) + jitter,
        values,
        s=17,
        facecolor=method_colors[method],
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=3,
    )
    ax_a.scatter(
        pos,
        float(np.mean(values)),
        marker="D",
        s=43,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.0,
        zorder=5,
    )

ax_a.set_xticks(positions, METHOD_ORDER_PANEL_A)
ax_a.set_ylim(0.55, 0.86)
ax_a.set_yticks(np.arange(0.55, 0.861, 0.05))
ax_a.set_ylabel("Best mean outer-fold ROC–AUC")
ax_a.set_title("Distribution across 31 source configurations", pad=8)
ax_a.grid(axis="y", alpha=0.24, linewidth=0.7)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.text(
    -0.13,
    1.04,
    "(a)",
    transform=ax_a.transAxes,
    fontsize=11,
    fontweight="bold",
    va="bottom",
)

mean_handle = Line2D(
    [0],
    [0],
    marker="D",
    linestyle="None",
    markerfacecolor="white",
    markeredgecolor="#111111",
    markersize=6,
    label="Mean",
)
ax_a.legend(handles=[mean_handle], loc="lower right", frameon=False)

# Panel (b)
x = np.arange(len(METHOD_ORDER_PANEL_B))
bottom = np.zeros(len(METHOD_ORDER_PANEL_B), dtype=float)
for level in FUSION_ORDER:
    values = win_counts.loc[METHOD_ORDER_PANEL_B, level].to_numpy(dtype=float)
    ax_b.bar(
        x,
        values,
        bottom=bottom,
        width=0.62,
        label=level,
        color=fusion_colors[level],
        edgecolor="white",
        linewidth=0.7,
    )
    bottom += values

totals = win_counts.loc[METHOD_ORDER_PANEL_B, "Total"].to_numpy(dtype=int)
for xpos, total in zip(x, totals):
    y = total + 0.45 if total > 0 else 0.35
    ax_b.text(xpos, y, str(total), ha="center", va="bottom", fontweight="bold")

ax_b.set_xticks(x, METHOD_ORDER_PANEL_B)
ax_b.set_ylabel("Number of configurations ranked first")
ax_b.set_ylim(0, 22.5)
ax_b.set_yticks(np.arange(0, 23, 5))
ax_b.set_title("Configuration-level wins by feature-selection strategy", pad=8)
ax_b.grid(axis="y", alpha=0.24, linewidth=0.7)
ax_b.set_axisbelow(True)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.text(
    -0.16,
    1.04,
    "(b)",
    transform=ax_b.transAxes,
    fontsize=11,
    fontweight="bold",
    va="bottom",
)
ax_b.legend(
    title="Fusion level",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=False,
    columnspacing=1.0,
    handlelength=1.2,
)

fig.subplots_adjust(bottom=0.23, top=0.90, left=0.08, right=0.99)
fig.savefig(FIGURE_PDF, bbox_inches="tight")
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
plt.close(fig)

print("\nSaved manuscript figure:")
print(f"  PDF: {FIGURE_PDF}")
print(f"  PNG: {FIGURE_PNG}")
print("\nSaved audit/supplementary tables in:")
print(f"  {AUDIT_DIR}")
