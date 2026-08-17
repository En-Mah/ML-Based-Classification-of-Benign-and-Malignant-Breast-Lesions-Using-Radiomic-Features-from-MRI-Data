#!/usr/bin/env python3
"""
Build the four-panel publication figure for the highest-ranked model.

Target pipeline
---------------
Dataset: ADC_Post1_Pre
Feature selection: Elastic Net -> mRMR
Feature count: Auto-k
SMOTE: No
Classifier: SVM_Linear

The script treats pooled_predictions.csv as the authoritative source and
recomputes the ROC curve, precision-recall curve, threshold-0.5 confusion
matrix, calibration curve, and all pooled metrics. Existing plot-data CSV
files are used for consistency checks only.

Expected input directory
------------------------
Triple/
nested_cv_outputs_all_triple_concat_batch_smote_compare/
ADC_Post1_Pre/
fs_method_comparison/
elasticnet_then_mrmr_auto_k_no_smote/
plot_data/

The script also supports plot_data/SVM_Linear/ and can search recursively
below the supplied run directory.

Outputs
-------
Figs/best_overall_model_oof_performance.pdf
Figs/best_overall_model_oof_performance.png
Figs/best_overall_model_oof_metrics.csv
Figs/best_overall_model_oof_metrics.json

Example
-------
python build_best_overall_model_oof_performance.py \
    --project-root "/path/to/project"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_RELATIVE_RUN_DIR = Path(
    "Triple"
) / "nested_cv_outputs_all_triple_concat_batch_smote_compare" / (
    "ADC_Post1_Pre"
) / "fs_method_comparison" / (
    "elasticnet_then_mrmr_auto_k_no_smote"
)

EXPECTED_MODEL = "SVM_Linear"
EXPECTED_REFERENCE = {
    "pooled_roc_auc": 0.799,
    "pooled_average_precision": 0.832,
    "pooled_balanced_accuracy": 0.713,
    "pooled_f1": 0.746,
}
REFERENCE_TOLERANCE = 0.006

TRUE_ALIASES = (
    "y_true",
    "true_label",
    "actual",
    "actual_label",
    "label",
    "target",
    "class",
)
PROB_ALIASES = (
    "y_prob",
    "y_probability",
    "probability",
    "predicted_probability",
    "positive_probability",
    "malignant_probability",
    "score",
    "y_score",
)
PRED_ALIASES = (
    "y_pred",
    "predicted_label",
    "prediction",
    "predicted_class",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the four-panel pooled out-of-fold evaluation figure "
            "for ADC_Post1_Pre / ElasticNet->mRMR / Auto-k / no SMOTE / "
            "SVM_Linear."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Direct path to the final run directory or its plot_data directory. "
            "When omitted, the expected Triple/... path is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <project-root>/Figs.",
    )
    parser.add_argument(
        "--model-name",
        default=EXPECTED_MODEL,
        help=f"Expected classifier name. Default: {EXPECTED_MODEL}.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Malignant-class probability threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=8,
        help="Number of calibration bins. Default: 8.",
    )
    parser.add_argument(
        "--calibration-strategy",
        choices=("quantile", "uniform"),
        default="quantile",
        help="Calibration binning strategy. Default: quantile.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600.",
    )
    parser.add_argument(
        "--strict-reference-check",
        action="store_true",
        help=(
            "Fail when recomputed key metrics differ materially from the "
            "manuscript reference values."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser.parse_args()


def norm_name(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_column(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    normalized = {norm_name(col): str(col) for col in df.columns}
    for alias in aliases:
        key = norm_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Could not find any of the required columns {tuple(aliases)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def map_binary_labels(values: pd.Series) -> np.ndarray:
    """Map common benign/malignant encodings to 0/1 without guessing."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int).to_numpy()

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        unique = set(numeric.astype(float).unique())
        if unique.issubset({0.0, 1.0}):
            return numeric.astype(int).to_numpy()

    mapping = {
        "0": 0,
        "benign": 0,
        "class0": 0,
        "negative": 0,
        "false": 0,
        "1": 1,
        "malignant": 1,
        "class1": 1,
        "positive": 1,
        "true": 1,
    }
    mapped = values.astype(str).map(lambda x: mapping.get(norm_name(x)))
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            "y_true must be binary and encode benign as 0 and malignant as 1. "
            f"Unrecognized values: {bad}"
        )
    return mapped.astype(int).to_numpy()


def read_predictions(plot_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    path = plot_dir / "pooled_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    true_col = find_column(df, TRUE_ALIASES)
    prob_col = find_column(df, PROB_ALIASES)
    pred_col = find_column(df, PRED_ALIASES, required=False)

    y_true = map_binary_labels(df[true_col])
    y_prob = pd.to_numeric(df[prob_col], errors="coerce").to_numpy(dtype=float)

    if len(y_true) == 0:
        raise ValueError("pooled_predictions.csv contains no observations.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both benign (0) and malignant (1) observations are required.")
    if not np.isfinite(y_prob).all():
        raise ValueError("The probability column contains NaN or infinite values.")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError(
            "Calibration and a threshold of 0.5 require probabilities in [0, 1]. "
            f"Observed range: [{y_prob.min():.6g}, {y_prob.max():.6g}]. "
            "Do not substitute raw SVM decision scores."
        )

    stored_pred = None
    if pred_col is not None:
        stored_pred = map_binary_labels(df[pred_col])

    return df, y_true, y_prob, stored_pred


def summary_mentions_model(plot_dir: Path, expected_model: str) -> bool:
    expected = norm_name(expected_model)

    # A model-specific directory name is strong evidence.
    if norm_name(plot_dir.name) == expected:
        return True

    # summary_statistics.csv is written for the exact predictions stored in
    # the same plot_data directory. If it names a different model, reject it.
    summary_path = plot_dir / "summary_statistics.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            for col in summary.columns:
                if norm_name(col) in {"model", "modelname", "classifier"}:
                    values = [norm_name(v) for v in summary[col].dropna()]
                    if values:
                        return values[0] == expected
        except Exception:
            pass

    # model_summary.csv normally contains all classifiers sorted by rank.
    # Only the first row can identify the champion predictions stored at the
    # root plot_data level; merely containing SVM_Linear is not sufficient.
    model_summary_path = plot_dir / "model_summary.csv"
    if model_summary_path.exists():
        try:
            summary = pd.read_csv(model_summary_path)
            for col in summary.columns:
                if norm_name(col) in {"model", "modelname", "classifier"}:
                    values = [norm_name(v) for v in summary[col].dropna()]
                    if values:
                        return values[0] == expected
        except Exception:
            pass

    return False


def resolve_plot_dir(
    project_root: Path,
    supplied_run_dir: Path | None,
    expected_model: str,
) -> Path:
    project_root = project_root.expanduser().resolve()

    if supplied_run_dir is None:
        base = project_root / DEFAULT_RELATIVE_RUN_DIR
    else:
        supplied_run_dir = supplied_run_dir.expanduser()
        base = supplied_run_dir if supplied_run_dir.is_absolute() else project_root / supplied_run_dir
        base = base.resolve()

    direct_candidates = [
        base,
        base / "plot_data",
        base / "plot_data" / expected_model,
        base / expected_model,
    ]
    valid_direct = [p for p in direct_candidates if (p / "pooled_predictions.csv").exists()]

    if valid_direct:
        model_matches = [p for p in valid_direct if summary_mentions_model(p, expected_model)]
        return model_matches[0] if model_matches else valid_direct[0]

    if not base.exists():
        raise FileNotFoundError(
            f"Run directory does not exist:\n{base}\n\n"
            "Pass --run-dir with the correct final run directory."
        )

    recursive = sorted(
        {p.parent for p in base.rglob("pooled_predictions.csv")},
        key=lambda p: (len(p.parts), str(p)),
    )
    if not recursive:
        raise FileNotFoundError(
            f"No pooled_predictions.csv was found below:\n{base}"
        )

    model_matches = [p for p in recursive if summary_mentions_model(p, expected_model)]
    if len(model_matches) == 1:
        return model_matches[0]
    if len(model_matches) > 1:
        exact_folder = [p for p in model_matches if norm_name(p.name) == norm_name(expected_model)]
        if len(exact_folder) == 1:
            return exact_folder[0]
        raise RuntimeError(
            "Multiple candidate plot-data directories appear to contain "
            f"{expected_model}:\n" + "\n".join(f"  - {p}" for p in model_matches)
        )

    if len(recursive) == 1:
        warnings.warn(
            f"Using the only pooled-prediction directory found, but its metadata "
            f"did not explicitly identify {expected_model}:\n{recursive[0]}"
        )
        return recursive[0]

    raise RuntimeError(
        "Multiple pooled-prediction directories were found and the target model "
        "could not be identified unambiguously:\n"
        + "\n".join(f"  - {p}" for p in recursive)
        + "\nPass --run-dir directly to the correct plot_data directory."
    )


def safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else math.nan


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics: dict[str, float | int] = {
        "n_observations": int(len(y_true)),
        "n_benign": int(np.sum(y_true == 0)),
        "n_malignant": int(np.sum(y_true == 1)),
        "malignant_prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "pooled_roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pooled_average_precision": float(average_precision_score(y_true, y_prob)),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "pooled_specificity": safe_specificity(y_true, y_pred),
        "pooled_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "pooled_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return metrics, cm, y_pred


def extract_fold_statistics(plot_dir: Path) -> dict[str, float | int]:
    path = plot_dir / "per_fold_metrics.csv"
    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return {}

    result: dict[str, float | int] = {}

    aliases = {
        "fold_auc": ("auc", "roc_auc", "fold_auc", "test_auc"),
        "fold_pr_auc": ("pr_auc", "average_precision", "fold_pr_auc"),
        "n_features": (
            "n_features",
            "selected_features",
            "n_selected_features",
            "feature_count",
        ),
    }

    for output_name, candidates in aliases.items():
        col = find_column(df, candidates, required=False)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        if output_name == "n_features":
            result.update(
                {
                    "feature_count_mean": float(values.mean()),
                    "feature_count_median": float(values.median()),
                    "feature_count_min": int(values.min()),
                    "feature_count_max": int(values.max()),
                }
            )
        else:
            result[f"{output_name}_mean"] = float(values.mean())
            result[f"{output_name}_sd"] = float(values.std(ddof=0))

    return result


def compare_stored_predictions(
    stored_pred: np.ndarray | None,
    threshold_pred: np.ndarray,
    threshold: float,
) -> None:
    if stored_pred is None:
        return
    disagreements = int(np.sum(stored_pred != threshold_pred))
    if disagreements:
        warnings.warn(
            f"The stored y_pred column differs from y_prob >= {threshold:g} for "
            f"{disagreements} observations. The figure uses the explicitly "
            f"requested threshold of {threshold:g}."
        )


def compare_stored_confusion_matrix(
    plot_dir: Path,
    computed_cm: np.ndarray,
) -> None:
    path = plot_dir / "confusion_matrix.csv"
    if not path.exists():
        return

    try:
        raw = pd.read_csv(path)
        numeric = raw.select_dtypes(include=[np.number]).to_numpy()
        if numeric.shape[1] > 2:
            numeric = numeric[:, -2:]
        if numeric.shape != (2, 2):
            warnings.warn(
                f"Could not interpret {path.name} as a 2x2 numeric matrix; "
                "the figure uses the matrix recomputed from pooled predictions."
            )
            return
        stored_cm = numeric.astype(int)
        if not np.array_equal(stored_cm, computed_cm):
            warnings.warn(
                "confusion_matrix.csv does not match the threshold-0.5 matrix "
                "recomputed from pooled_predictions.csv. The recomputed matrix "
                "is used in the figure."
            )
    except Exception as exc:
        warnings.warn(f"Could not cross-check {path}: {exc}")


def compare_curve_files(
    plot_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    roc_path = plot_dir / "roc_curve.csv"
    if roc_path.exists():
        try:
            stored = pd.read_csv(roc_path)
            fpr_col = find_column(stored, ("fpr", "false_positive_rate"))
            tpr_col = find_column(stored, ("tpr", "true_positive_rate"))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            if len(stored) != len(fpr):
                warnings.warn(
                    "roc_curve.csv has a different number of rows from the "
                    "curve recomputed from pooled_predictions.csv."
                )
            else:
                delta = max(
                    np.max(np.abs(pd.to_numeric(stored[fpr_col]) - fpr)),
                    np.max(np.abs(pd.to_numeric(stored[tpr_col]) - tpr)),
                )
                if delta > 1e-10:
                    warnings.warn(
                        "roc_curve.csv differs from the recomputed empirical ROC curve."
                    )
        except Exception as exc:
            warnings.warn(f"Could not cross-check {roc_path}: {exc}")

    pr_path = plot_dir / "pr_curve.csv"
    if pr_path.exists():
        try:
            stored = pd.read_csv(pr_path)
            precision_col = find_column(stored, ("precision",))
            recall_col = find_column(stored, ("recall",))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            if len(stored) != len(precision):
                warnings.warn(
                    "pr_curve.csv has a different number of rows from the "
                    "curve recomputed from pooled_predictions.csv."
                )
            else:
                delta = max(
                    np.max(np.abs(pd.to_numeric(stored[precision_col]) - precision)),
                    np.max(np.abs(pd.to_numeric(stored[recall_col]) - recall)),
                )
                if delta > 1e-10:
                    warnings.warn(
                        "pr_curve.csv differs from the recomputed empirical PR curve."
                    )
        except Exception as exc:
            warnings.warn(f"Could not cross-check {pr_path}: {exc}")


def validate_reference_values(
    metrics: dict[str, float | int],
    strict: bool,
) -> None:
    mismatches: list[str] = []
    for key, expected in EXPECTED_REFERENCE.items():
        actual = float(metrics[key])
        if abs(actual - expected) > REFERENCE_TOLERANCE:
            mismatches.append(
                f"{key}: recomputed={actual:.4f}, manuscript reference={expected:.4f}"
            )

    if mismatches:
        message = (
            "The selected data do not reproduce one or more verified reference "
            "metrics within the rounding tolerance:\n  - "
            + "\n  - ".join(mismatches)
            + "\nCheck that the input corresponds specifically to "
            "ADC_Post1_Pre / elasticnet_then_mrmr_auto_k_no_smote / SVM_Linear."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_confusion_matrix_panel(
    ax: plt.Axes,
    cm: np.ndarray,
    threshold: float,
) -> None:
    total = int(cm.sum())
    cmap = plt.get_cmap("Blues")
    image = ax.imshow(cm, cmap=cmap, aspect="equal")

    ax.set_title(f"Confusion matrix (threshold = {threshold:g})", pad=8)
    ax.set_xticks([0, 1], labels=["Predicted benign", "Predicted malignant"])
    ax.set_yticks([0, 1], labels=["True benign", "True malignant"])
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    threshold_color = (float(cm.max()) + float(cm.min())) / 2.0
    for row in range(2):
        for col in range(2):
            count = int(cm[row, col])
            percent = 100.0 * count / total if total else math.nan
            text_color = "white" if count > threshold_color else "black"
            ax.text(
                col,
                row,
                f"{count}\n({percent:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
                linespacing=1.25,
            )

    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    for row in range(2):
        for col in range(2):
            ax.text(
                col - 0.43,
                row - 0.40,
                cell_names[row, col],
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=(
                    "white"
                    if cm[row, col] > threshold_color
                    else "black"
                ),
            )

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Keep the color mapping stable and suppress an unnecessary colorbar.
    image.set_norm(colors.Normalize(vmin=0, vmax=max(1, int(cm.max()))))


def build_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    metrics: dict[str, float | int],
    output_pdf: Path,
    output_png: Path,
    threshold: float,
    calibration_bins: int,
    calibration_strategy: str,
    dpi: int,
    show: bool,
) -> None:
    configure_matplotlib()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=calibration_bins,
        strategy=calibration_strategy,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        constrained_layout=False,
    )
    ax_roc, ax_pr, ax_cm, ax_cal = axes.ravel()

    # Panel (a): ROC
    ax_roc.plot(
        fpr,
        tpr,
        drawstyle="steps-post",
        linewidth=2.2,
    )
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="0.45",
    )
    ax_roc.set_title("Pooled out-of-fold ROC curve", pad=8)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
#    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax_roc.text(
        0.97,
        0.05,
        f"Pooled ROC-AUC = {metrics['pooled_roc_auc']:.3f}",
        transform=ax_roc.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.8"},
    )
    panel_label(ax_roc, "(a)")

    # Panel (b): PR
    ax_pr.plot(
        recall,
        precision,
        drawstyle="steps-post",
        linewidth=2.2,
        label="Linear SVM",
    )
    ax_pr.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.2,
        color="0.45",
        label=f"Malignant prevalence = {prevalence:.3f}",
    )
    ax_pr.set_title("Pooled precision-recall curve", pad=8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax_pr.legend(loc="lower left", frameon=False)
    ax_pr.text(
        0.97,
        0.05,
        f"Average precision = {metrics['pooled_average_precision']:.3f}",
        transform=ax_pr.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.8"},
    )
    panel_label(ax_pr, "(b)")

    # Panel (c): confusion matrix
    plot_confusion_matrix_panel(ax_cm, cm, threshold)
    panel_label(ax_cm, "(c)")

    # Panel (d): calibration
    ax_cal.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="0.45",
        label="Perfect calibration",
    )
    ax_cal.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        label="Linear SVM",
    )
    ax_cal.set_title("Pooled probability-calibration curve", pad=8)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed malignant proportion")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1.02)
#    ax_cal.set_aspect("equal", adjustable="box")
    ax_cal.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax_cal.legend(loc="upper left", frameon=False)
    ax_cal.text(
        0.97,
        0.05,
        f"Brier score = {metrics['brier_score']:.3f}",
        transform=ax_cal.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.8"},
    )
    panel_label(ax_cal, "(d)")

    fig.suptitle(
        "ADC-Post1-Pre | Elastic Net $\\rightarrow$ mRMR | Auto-$k$ | "
        "No SMOTE | Linear SVM",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.08,
        top=0.92,
        wspace=0.15,
        hspace=0.27,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def write_metrics(
    metrics: dict[str, float | int],
    fold_stats: dict[str, float | int],
    output_csv: Path,
    output_json: Path,
    source_dir: Path,
) -> None:
    combined = {
        "source_plot_data_directory": str(source_dir),
        "dataset": "ADC_Post1_Pre",
        "feature_selection": "elasticnet_then_mrmr_auto_k",
        "smote": "No",
        "classifier": EXPECTED_MODEL,
        **metrics,
        **fold_stats,
    }

    pd.DataFrame([combined]).to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)


def print_report(
    plot_dir: Path,
    output_pdf: Path,
    output_png: Path,
    output_csv: Path,
    output_json: Path,
    metrics: dict[str, float | int],
    fold_stats: dict[str, float | int],
) -> None:
    print("\n" + "=" * 78)
    print("BEST OVERALL MODEL - DATA-DERIVED OOF FIGURE")
    print("=" * 78)
    print(f"Input plot-data directory : {plot_dir}")
    print(f"Observations              : {metrics['n_observations']}")
    print(f"Benign / malignant        : {metrics['n_benign']} / {metrics['n_malignant']}")
    print(f"Pooled ROC-AUC            : {metrics['pooled_roc_auc']:.6f}")
    print(f"Pooled average precision  : {metrics['pooled_average_precision']:.6f}")
    print(f"Pooled balanced accuracy  : {metrics['pooled_balanced_accuracy']:.6f}")
    print(f"Pooled F1                 : {metrics['pooled_f1']:.6f}")
    print(
        "Confusion matrix         : "
        f"TN={metrics['TN']}, FP={metrics['FP']}, "
        f"FN={metrics['FN']}, TP={metrics['TP']}"
    )
    print(f"Brier score               : {metrics['brier_score']:.6f}")

    if fold_stats:
        print("\nFold-level values detected:")
        for key, value in fold_stats.items():
            if isinstance(value, float):
                print(f"  {key:28s}: {value:.6f}")
            else:
                print(f"  {key:28s}: {value}")

    print("\nSaved:")
    print(f"  PDF  : {output_pdf}")
    print(f"  PNG  : {output_png}")
    print(f"  CSV  : {output_csv}")
    print(f"  JSON : {output_json}")
    print("=" * 78)


def main() -> int:
    args = parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be strictly between 0 and 1.")
    if args.calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2.")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72.")

    project_root = args.project_root.expanduser().resolve()
    if args.output_dir is None:
        output_dir = project_root / "Figs"
    else:
        supplied_output = args.output_dir.expanduser()
        output_dir = (
            supplied_output.resolve()
            if supplied_output.is_absolute()
            else (project_root / supplied_output).resolve()
        )

    plot_dir = resolve_plot_dir(
        project_root=project_root,
        supplied_run_dir=args.run_dir,
        expected_model=args.model_name,
    )

    _, y_true, y_prob, stored_pred = read_predictions(plot_dir)
    metrics, cm, y_pred = compute_metrics(y_true, y_prob, args.threshold)

    compare_stored_predictions(stored_pred, y_pred, args.threshold)
    compare_stored_confusion_matrix(plot_dir, cm)
    compare_curve_files(plot_dir, y_true, y_prob)
    validate_reference_values(metrics, args.strict_reference_check)

    fold_stats = extract_fold_statistics(plot_dir)

    output_pdf = output_dir / "best_overall_model_oof_performance.pdf"
    output_png = output_dir / "best_overall_model_oof_performance.png"
    output_csv = output_dir / "best_overall_model_oof_metrics.csv"
    output_json = output_dir / "best_overall_model_oof_metrics.json"

    build_figure(
        y_true=y_true,
        y_prob=y_prob,
        cm=cm,
        metrics=metrics,
        output_pdf=output_pdf,
        output_png=output_png,
        threshold=args.threshold,
        calibration_bins=args.calibration_bins,
        calibration_strategy=args.calibration_strategy,
        dpi=args.dpi,
        show=args.show,
    )

    write_metrics(
        metrics=metrics,
        fold_stats=fold_stats,
        output_csv=output_csv,
        output_json=output_json,
        source_dir=plot_dir,
    )

    print_report(
        plot_dir=plot_dir,
        output_pdf=output_pdf,
        output_png=output_png,
        output_csv=output_csv,
        output_json=output_json,
        metrics=metrics,
        fold_stats=fold_stats,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
