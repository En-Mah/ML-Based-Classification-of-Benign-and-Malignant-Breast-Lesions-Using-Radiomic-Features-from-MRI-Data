#!/usr/bin/env python3
"""
Build the publication figure:
    Figs/best_model_feature_stability_and_shap.pdf

for the highest-ranked ADC--Post1--Pre pipeline:
    Elastic Net -> mRMR -> Auto-k -> no SMOTE -> Linear SVM.

The script deliberately uses the *existing nested-CV run artifacts* for
feature-selection recurrence and hyperparameter aggregation. It does NOT
re-run or overwrite the nested-CV performance analysis.

Panels
------
(a) Outer-fold selection frequency for recurrent features (selected >=2/5 folds)
(b) Global SHAP importance, mean(|SHAP|), for the same recurrent features
(c) SHAP beeswarm summary; each point is one ROI and colour is the standardized
    feature value.

Interpretation model
--------------------
A single Linear SVM is refitted on the full cohort *for interpretation only*:
  - feature space = recurrent features from the five outer folds;
  - median imputation = fitted on the full cohort;
  - standardization = fitted on the full cohort;
  - SVM C = median of the five outer-fold Optuna-selected C values;
  - no SMOTE;
  - by default, SHAP explains the Linear SVM class-1/malignant probability,
    matching the current Methods text; --shap-output decision can be used to
    explain the raw linear decision score instead.

The full-cohort model is never used to report validation performance.
Reported performance remains the nested out-of-fold performance stored in
nested_cv_results.pkl.

Expected run-artifact files
---------------------------
<run_dir>/df_selected_features_long.csv
<run_dir>/nested_cv_results.pkl
Optional:
<run_dir>/df_summary.csv
<run_dir>/effective_config.json or run_metadata.json

Example
-------
python build_best_model_feature_stability_and_shap.py \
  --dataset Triple/output/all_10_triple_concat_datasets/ADC_Post1_Pre_M1_triple.csv \
  --run-dir Triple/batch_all_10_triple_runner_artifacts_smote_compare/ADC_Post1_Pre/fs_method_comparison/elasticnet_then_mrmr_auto_k_no_smote \
  --output-dir Figs

If your selected run directory has a slightly different name, point --run-dir
at the directory that contains df_selected_features_long.csv and
nested_cv_results.pkl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import textwrap
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# -----------------------------------------------------------------------------
# Publication/reference values used ONLY to protect against selecting the wrong
# run directory (especially the all-five SHAP run).
# -----------------------------------------------------------------------------
REFERENCE = {
    "n_outer_folds": 5,
    "avg_features": 9.6,
    "n_unique_features": 28,
    "max_recurrence": 4,
    "mean_auc": 0.836,
    "pooled_auc": 0.799,
}

MODEL_NAME = "SVM_Linear"
MIN_RECURRENT_FOLDS = 2
RANDOM_STATE = 100


# -----------------------------------------------------------------------------
# Helpers: serialization, naming, metadata
# -----------------------------------------------------------------------------
def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def clean_feature_name(col_name: str, target_column: str = "Label") -> str:
    """Replicate the feature-name cleaning used in the analysis notebooks."""
    original = str(col_name).strip()
    protected = {
        target_column,
        "INFO_PatientName",
        "INFO_NameOfRoi",
        "PatientID",
    }
    if original in protected:
        return original

    name = original
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"MORPHOLOGICAL", "Morph", name, flags=re.IGNORECASE)
    name = re.sub(r"INTENSITY-BASED", "IB", name, flags=re.IGNORECASE)
    name = re.sub(r"INTENSITY-HISTOGRAM", "IH", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"^_+|_+$", "", name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def make_unique_columns(columns: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            out.append(col)
        else:
            seen[col] += 1
            out.append(f"{col}_{seen[col]}")
    return out


def sanitize_for_match(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", str(name))).strip("_").lower()


def wrap_label(name: str, width: int = 28) -> str:
    # Keep the scientifically meaningful feature name, but make it readable.
    return textwrap.fill(str(name), width=width, break_long_words=False, break_on_hyphens=False)





def short_display_label(feature_name: str) -> str:
    s = str(feature_name)

    # Source
    if s.startswith("ADC_"):
        source = "ADC"
        body = s[len("ADC_"):]
    elif s.startswith("Post1_"):
        source = "P1"
        body = s[len("Post1_"):]
    elif s.startswith("Pre_"):
        source = "Pre"
        body = s[len("Pre_"):]
    elif s.startswith("Post2_"):
        source = "P2"
        body = s[len("Post2_"):]
    elif s.startswith("T2_"):
        source = "T2"
        body = s[len("T2_"):]
    else:
        source = ""
        body = s

    replacements = {
        "IH_IntensityHistogramMedian_Intensity": "IH median",
        "IB_MedianIntensity": "IB median",
        "IH_MinimumHistogramGradientGreyLevel_Intensity": "IH min-grad",
        "IH_IntensityHistogramMode_Intensity": "IH mode",
        "Morph_Compactness1": "Comp.1",
        "Morph_Compactness2": "Comp.2",
        "Morph_Sphericity": "Sphericity",
        "GLSZM_NormalisedZoneSizeNonUniformity": "Norm ZSNU",
        "GLSZM_GreyLevelVariance": "GL var",
        "GLSZM_ZoneSizeVariance": "ZS var",
        "GLSZM_GreyLevelNonUniformity": "GL non-unif.",
        "GLSZM_LargeZoneLowGreyLevelEmphasis": "LZ low-GL emph.",
    }

    label = replacements.get(body, body.replace("_", " "))

    if source:
        return f"{source} | {label}"
    return label
    
    
    

def infer_source(feature_name: str) -> str:
    s = str(feature_name)
    for source in ("ADC", "Post1", "Pre", "Post2", "T2"):
        if re.search(rf"(^|_){re.escape(source)}(_|$)", s, flags=re.IGNORECASE):
            return source
    # Common modality prefix variants after sanitization
    lower = s.lower()
    if lower.startswith("adc_"):
        return "ADC"
    if lower.startswith("post1_"):
        return "Post1"
    if lower.startswith("pre_"):
        return "Pre"
    return "Unknown"


def infer_family(feature_name: str) -> str:
    s = str(feature_name)
    # Remove source prefix before family detection.
    s = re.sub(r"^(ADC|Post1|Pre|Post2|T2)_", "", s, flags=re.IGNORECASE)
    upper = s.upper()
    if upper.startswith("MORPH") or "MORPHOLOG" in upper:
        return "Morphology"
    if upper.startswith("IB_") or upper.startswith("INTENSITY_BASED"):
        return "Intensity"
    if upper.startswith("IH_") or upper.startswith("INTENSITY_HISTOGRAM"):
        return "Intensity histogram"
    for family in ("GLCM", "GLRLM", "GLSZM", "NGTDM", "NGLDM", "GLDZM"):
        if family in upper:
            return family
    return "Other"


def load_effective_config(run_dir: Path) -> Dict[str, Any]:
    cfg = run_dir / "effective_config.json"
    meta = run_dir / "run_metadata.json"
    if cfg.exists():
        with cfg.open("r", encoding="utf-8") as f:
            return json.load(f)
    if meta.exists():
        with meta.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("config", {}) or {}
    return {}


def aggregate_best_params(params_per_fold: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Median for numeric parameters; mode for categorical parameters."""
    if not params_per_fold:
        return {}
    valid = [p for p in params_per_fold if isinstance(p, dict)]
    if not valid:
        return {}

    keys = sorted(set().union(*(p.keys() for p in valid)))
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [p[key] for p in valid if key in p]
        if not vals:
            continue
        numeric = all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in vals
        )
        if numeric:
            med = float(np.median(vals))
            all_int = all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals)
            out[key] = int(round(med)) if all_int else med
        else:
            out[key] = Counter(vals).most_common(1)[0][0]
    return out


# -----------------------------------------------------------------------------
# Load and validate the selected nested-CV run
# -----------------------------------------------------------------------------
def load_run_artifacts(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    required = [
        run_dir / "df_selected_features_long.csv",
        run_dir / "nested_cv_results.pkl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Selected run directory is missing required artifact(s):\n  " + "\n  ".join(missing)
        )

    stability_long = pd.read_csv(run_dir / "df_selected_features_long.csv")
    with (run_dir / "nested_cv_results.pkl").open("rb") as f:
        nested_cv_results = pickle.load(f)
    config = load_effective_config(run_dir)
    return stability_long, nested_cv_results, config


def get_feature_column(df: pd.DataFrame) -> str:
    for c in ("Feature_Name", "Feature"):
        if c in df.columns:
            return c
    raise ValueError(
        "df_selected_features_long.csv must contain Feature_Name or Feature. "
        f"Found: {list(df.columns)}"
    )


def recurrence_table(stability_long: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    feature_col = get_feature_column(stability_long)
    if "Fold" not in stability_long.columns:
        raise ValueError("df_selected_features_long.csv does not contain a Fold column.")

    tmp = stability_long[["Fold", feature_col]].dropna().copy()
    # Defensive: a feature should count at most once per fold.
    tmp = tmp.drop_duplicates(subset=["Fold", feature_col])

    folds = sorted(pd.unique(tmp["Fold"]))
    n_folds = len(folds)
    counts = tmp[feature_col].value_counts()

    rec = counts.rename_axis("Feature_Name").reset_index(name="Selection_Count")
    rec["Selection_Frequency"] = rec["Selection_Count"] / n_folds
    rec["Recurrent"] = rec["Selection_Count"] >= MIN_RECURRENT_FOLDS
    rec["Source"] = rec["Feature_Name"].map(infer_source)
    rec["Feature_Family"] = rec["Feature_Name"].map(infer_family)
    rec = rec.sort_values(
        ["Selection_Count", "Feature_Name"], ascending=[False, True]
    ).reset_index(drop=True)

    avg_features_per_fold = float(len(tmp) / n_folds)
    return rec, n_folds, avg_features_per_fold


def compute_oof_metrics(nested_cv_results: Dict[str, Any], model_name: str) -> Dict[str, float]:
    if model_name not in nested_cv_results:
        raise KeyError(
            f"{model_name!r} not found in nested_cv_results. "
            f"Available models: {sorted(nested_cv_results.keys())}"
        )
    res = nested_cv_results[model_name]
    y_true = np.asarray(res["y_true"]).astype(int)
    y_prob = np.asarray(res["y_prob"], dtype=float)
    y_pred = np.asarray(res["y_pred"]).astype(int)
    fold_aucs = np.asarray(res.get("fold_aucs", []), dtype=float)

    return {
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs.size else float("nan"),
        "std_auc": float(np.std(fold_aucs, ddof=1)) if fold_aucs.size > 1 else float("nan"),
        "pooled_auc": float(roc_auc_score(y_true, y_prob)),
        "pooled_pr_auc": float(average_precision_score(y_true, y_prob)),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def validate_reference_run(
    rec: pd.DataFrame,
    n_folds: int,
    avg_features_per_fold: float,
    oof: Dict[str, float],
    config: Dict[str, Any],
    allow_mismatch: bool,
) -> None:
    problems: List[str] = []

    if n_folds != REFERENCE["n_outer_folds"]:
        problems.append(f"outer folds = {n_folds}, expected 5")
    if not math.isclose(avg_features_per_fold, REFERENCE["avg_features"], abs_tol=0.11):
        problems.append(
            f"average selected features/fold = {avg_features_per_fold:.3f}, expected 9.6"
        )
    if len(rec) != REFERENCE["n_unique_features"]:
        problems.append(f"unique selected features = {len(rec)}, expected 28")
    if int(rec["Selection_Count"].max()) != REFERENCE["max_recurrence"]:
        problems.append(
            f"maximum recurrence = {int(rec['Selection_Count'].max())}, expected 4"
        )
    if not math.isclose(oof["mean_auc"], REFERENCE["mean_auc"], abs_tol=0.015):
        problems.append(f"Linear-SVM mean AUC = {oof['mean_auc']:.3f}, expected about 0.836")
    if not math.isclose(oof["pooled_auc"], REFERENCE["pooled_auc"], abs_tol=0.015):
        problems.append(f"Linear-SVM pooled AUC = {oof['pooled_auc']:.3f}, expected about 0.799")

    fs_method = str(config.get("FS_METHOD", "")).lower()
    if fs_method and fs_method != "elasticnet_then_mrmr":
        problems.append(f"FS_METHOD={config.get('FS_METHOD')!r}, expected elasticnet_then_mrmr")

    use_smote = config.get("USE_SMOTE", None)
    if use_smote not in (None, False, 0, "False", "false", "0"):
        problems.append(f"USE_SMOTE={use_smote!r}, expected no SMOTE")

    if problems:
        message = (
            "\nREFERENCE-RUN CHECK FAILED. This may be the wrong run directory "
            "(for example an all-five run):\n  - " + "\n  - ".join(problems)
        )
        if allow_mismatch:
            warnings.warn(message)
        else:
            raise RuntimeError(message + "\nUse --allow-reference-mismatch only if the mismatch is intentional.")


# -----------------------------------------------------------------------------
# Dataset loading and feature alignment
# -----------------------------------------------------------------------------
def load_dataset(dataset_path: Path, label_col: str, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)

    # Same broad cleaning behavior as the notebooks.
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)

    cleaned = [clean_feature_name(c, label_col) for c in df.columns]
    df.columns = make_unique_columns(cleaned)

    if group_col not in df.columns and "INFO_PatientName" in df.columns:
        df = df.rename(columns={"INFO_PatientName": group_col})

    if label_col not in df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found after cleaning. Columns include: {list(df.columns)[:20]} ..."
        )
    if group_col not in df.columns:
        warnings.warn(
            f"Group column {group_col!r} not found. This does not affect the full-cohort XAI refit, "
            "but PatientID will be unavailable in the long SHAP output."
        )

    # Convert +/-inf to missing; imputation occurs below.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def align_recurrent_features(df: pd.DataFrame, recurrent: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    # Exact match first; sanitized fallback second.
    by_sanitized: Dict[str, List[str]] = {}
    for col in df.columns:
        by_sanitized.setdefault(sanitize_for_match(col), []).append(col)

    matched: List[str] = []
    mapping: Dict[str, str] = {}
    unmatched: List[str] = []
    ambiguous: List[Tuple[str, List[str]]] = []

    for feat in recurrent["Feature_Name"].tolist():
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            matched.append(feat)
            mapping[feat] = feat
            continue

        candidates = [
            c for c in by_sanitized.get(sanitize_for_match(feat), [])
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(candidates) == 1:
            matched.append(candidates[0])
            mapping[feat] = candidates[0]
        elif len(candidates) > 1:
            ambiguous.append((feat, candidates))
        else:
            unmatched.append(feat)

    if ambiguous:
        msg = "\n".join(f"  {f}: {c}" for f, c in ambiguous)
        raise ValueError(f"Ambiguous recurrent-feature matches:\n{msg}")
    if unmatched:
        msg = "\n".join(f"  - {f}" for f in unmatched)
        raise ValueError(
            "Some recurrent features from the nested-CV run were not found in the supplied dataset:\n"
            + msg
            + "\nUse the exact ADC--Post1--Pre triple dataset used by the selected run."
        )

    # Avoid accidental duplicate mapping.
    if len(set(matched)) != len(matched):
        raise ValueError("Two recurrent features mapped to the same dataset column; aborting.")

    return matched, mapping


# -----------------------------------------------------------------------------
# Interpretation model and SHAP
# -----------------------------------------------------------------------------
def fit_interpretation_model(
    df: pd.DataFrame,
    feature_columns: List[str],
    y: np.ndarray,
    nested_cv_results: Dict[str, Any],
    model_name: str,
) -> Tuple[SVC, SimpleImputer, StandardScaler, pd.DataFrame, Dict[str, Any]]:
    X_raw = df[feature_columns].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    # Defensive zero-variance removal. It should normally keep every recurrent feature.
    vt = VarianceThreshold(threshold=1e-10)
    X_vt = vt.fit_transform(X_imp)
    keep = vt.get_support()
    if not np.all(keep):
        removed = [f for f, k in zip(feature_columns, keep) if not k]
        raise RuntimeError(
            "A recurrent feature became zero-variance on the full cohort, so the publication feature space "
            f"would change. Removed features: {removed}"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vt)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)

    best_params_per_fold = nested_cv_results[model_name].get("best_params_per_fold", [])
    aggregated = aggregate_best_params(best_params_per_fold)
    if "C" not in aggregated:
        raise RuntimeError(
            "Could not obtain Linear-SVM C from nested_cv_results['SVM_Linear']['best_params_per_fold']."
        )

    model = SVC(
        C=float(aggregated["C"]),
        kernel="linear",
        probability=True,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_scaled_df, y)
    return model, imputer, scaler, X_scaled_df, aggregated


def compute_shap(
    model: SVC,
    X_scaled_df: pd.DataFrame,
    output_mode: str = "probability",
) -> shap.Explanation:
    """Compute SHAP values for the interpretation model.

    probability (default)
        Explain predict_proba[:, 1], matching the current Methods rule that
        classifiers supporting predict_proba are interpreted on class-1
        probability. This may be slower because the Platt-scaled probability
        is nonlinear even when the SVM kernel is linear.

    decision
        Explain the raw linear SVM decision function using LinearExplainer.
        This is exact and faster, but should only be used if the Methods text is
        changed accordingly.
    """
    if output_mode == "decision":
        explainer = shap.LinearExplainer(model, X_scaled_df)
        explanation = explainer(X_scaled_df)
    elif output_mode == "probability":
        def positive_probability(X_input):
            X_input = pd.DataFrame(X_input, columns=X_scaled_df.columns)
            return model.predict_proba(X_input)[:, 1]

        masker = shap.maskers.Independent(
            X_scaled_df,
            max_samples=X_scaled_df.shape[0],
        )
        explainer = shap.Explainer(
            positive_probability,
            masker,
            seed=RANDOM_STATE,
        )
        explanation = explainer(X_scaled_df)
    else:
        raise ValueError(f"Unsupported SHAP output mode: {output_mode!r}")

    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
        explanation = shap.Explanation(
            values=values,
            base_values=np.asarray(explanation.base_values).squeeze(),
            data=np.asarray(explanation.data),
            feature_names=list(X_scaled_df.columns),
        )
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")
    return explanation


def build_feature_results(
    recurrence: pd.DataFrame,
    recurrent_original_names: List[str],
    feature_mapping: Dict[str, str],
    X_scaled_df: pd.DataFrame,
    explanation: shap.Explanation,
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(explanation.values), axis=0)

    # Map run-feature name -> fitted dataset column.
    inv_mapping = {v: k for k, v in feature_mapping.items()}
    rec_lookup = recurrence.set_index("Feature_Name")

    rows: List[Dict[str, Any]] = []
    for j, dataset_col in enumerate(X_scaled_df.columns):
        run_feat = inv_mapping[dataset_col]
        xj = X_scaled_df.iloc[:, j].to_numpy(dtype=float)
        sj = explanation.values[:, j].astype(float)
        rho, p = spearmanr(xj, sj)
        if np.isnan(rho):
            direction = "undetermined"
        elif rho > 0:
            direction = "higher values -> malignant"
        elif rho < 0:
            direction = "higher values -> benign"
        else:
            direction = "no monotonic direction"

        rows.append({
            "Feature_Name": run_feat,
            "Dataset_Column": dataset_col,
            "Selection_Count": int(rec_lookup.loc[run_feat, "Selection_Count"]),
            "Selection_Frequency": float(rec_lookup.loc[run_feat, "Selection_Frequency"]),
            "Source": infer_source(run_feat),
            "Feature_Family": infer_family(run_feat),
            "MeanAbsSHAP": float(mean_abs[j]),
            "Spearman_rho_standardized_value_vs_SHAP": float(rho) if not np.isnan(rho) else np.nan,
            "Spearman_p_value": float(p) if not np.isnan(p) else np.nan,
            "SHAP_Direction": direction,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def make_figure(
    feature_results: pd.DataFrame,
    X_scaled_df: pd.DataFrame,
    explanation: shap.Explanation,
    output_dir: Path,
    shap_output_mode: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Publication labels are intentionally shorter than the original feature names.
    # This changes presentation only; all saved data tables retain the full names.
    display_name = {
        row.Feature_Name: short_display_label(row.Feature_Name)
        for row in feature_results.itertuples()
    }

    # The fitted columns may differ very slightly from run feature names after cleaning.
    dataset_to_run = dict(zip(feature_results["Dataset_Column"], feature_results["Feature_Name"]))
    shap_feature_names = [display_name[dataset_to_run[c]] for c in X_scaled_df.columns]
    explanation_plot = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=X_scaled_df.to_numpy(),
        feature_names=shap_feature_names,
    )

    n_features = len(feature_results)

    # Sized for a full-width two-column manuscript figure. The PDF/SVG outputs are
    # vector graphics, so they remain sharp when scaled by LaTeX.
    fig_height = max(6.0, min(8.0, 3.7 + 0.28 * n_features))
    fig = plt.figure(figsize=(13.2, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.08, 1.10, 1.72],
        wspace=0.54,
        left=0.075,
        right=0.955,
        top=0.91,
        bottom=0.13,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    title_fs = 11.0
    axis_fs = 9.6
    tick_fs = 8.0

    # ---- Panel (a): recurrence ------------------------------------------------
    a = feature_results.sort_values(
        ["Selection_Frequency", "MeanAbsSHAP", "Feature_Name"],
        ascending=[True, True, False]
    ).copy()
    y_a = np.arange(len(a))
    ax_a.barh(y_a, a["Selection_Frequency"].to_numpy())
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels([display_name[x] for x in a["Feature_Name"]], fontsize=tick_fs)
    ax_a.set_xlim(0, 1.0)
    ax_a.xaxis.set_major_locator(FixedLocator([0.4, 0.6, 0.8]))
    ax_a.set_xlabel("Outer-fold selection frequency", fontsize=axis_fs)
    ax_a.set_title("Feature-selection recurrence", fontsize=title_fs, fontweight="bold", pad=9)
    ax_a.grid(axis="x", linestyle=":", alpha=0.30)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(axis="x", labelsize=tick_fs)
    ax_a.tick_params(axis="y", length=0, pad=3)
    for i, row in enumerate(a.itertuples()):
        ax_a.text(
            min(row.Selection_Frequency + 0.018, 0.94), i,
            f"{int(row.Selection_Count)}/5",
            va="center", ha="left", fontsize=7.8
        )

    # ---- Panel (b): global SHAP ----------------------------------------------
    b = feature_results.sort_values(
        ["MeanAbsSHAP", "Feature_Name"], ascending=[True, False]
    ).copy()
    y_b = np.arange(len(b))
    ax_b.barh(y_b, b["MeanAbsSHAP"].to_numpy())
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([display_name[x] for x in b["Feature_Name"]], fontsize=tick_fs)
    ax_b.set_xlabel("Mean absolute SHAP value", fontsize=axis_fs)
    ax_b.set_title("Global SHAP importance", fontsize=title_fs, fontweight="bold", pad=9)
    ax_b.grid(axis="x", linestyle=":", alpha=0.30)
    ax_b.set_axisbelow(True)
    ax_b.tick_params(axis="x", labelsize=tick_fs)
    ax_b.tick_params(axis="y", length=0, pad=3)

    # ---- Panel (c): SHAP beeswarm --------------------------------------------
    # SHAP's beeswarm sorts by mean absolute SHAP by default.
    shap.plots.beeswarm(
        explanation_plot,
        max_display=n_features,
        ax=ax_c,
        show=False,
        color_bar=True,
        plot_size=None,
        s=16,
    )
    ax_c.axvline(0.0, linewidth=1.05, linestyle="--", color="0.35", zorder=0)
    if shap_output_mode == "probability":
        ax_c.set_xlabel("SHAP value (malignant-class probability)", fontsize=axis_fs)
    else:
        ax_c.set_xlabel("SHAP value (malignant-class decision score)", fontsize=axis_fs)
    ax_c.set_title("SHAP summary", fontsize=title_fs, fontweight="bold", pad=9)
    ax_c.tick_params(axis="y", labelsize=tick_fs, length=0, pad=3)
    ax_c.tick_params(axis="x", labelsize=tick_fs)
    ax_c.grid(axis="x", linestyle=":", alpha=0.22)
    ax_c.set_axisbelow(True)

    # SHAP creates its own colorbar axis. Make it shorter and align it vertically
    # with the beeswarm panel rather than letting it dominate the composition.
    extra_axes = [ax for ax in fig.axes if ax not in (ax_a, ax_b, ax_c)]
    if extra_axes:
        cbar_ax = extra_axes[-1]
        pos = ax_c.get_position()
        cbar_width = 0.010
        cbar_gap = 0.012
        cbar_height = pos.height * 0.82
        cbar_y = pos.y0 + (pos.height - cbar_height) / 2
        cbar_ax.set_position([pos.x1 + cbar_gap, cbar_y, cbar_width, cbar_height])
        cbar_ax.tick_params(labelsize=8.0, length=0)
        cbar_ax.set_ylabel("Feature value", fontsize=9.0, labelpad=8)

    # Panel labels: aligned consistently above the upper-left corner of each panel.
    for ax, label in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        ax.text(
            -0.12, 1.035, label,
            transform=ax.transAxes,
            fontsize=12.0,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    stem = output_dir / "best_model_feature_stability_and_shap"
    pdf = str(stem.with_suffix(".pdf"))
    png = str(stem.with_suffix(".png"))
    svg = str(stem.with_suffix(".svg"))

    # Vector PDF/SVG are preferred for the manuscript; PNG is a 600-dpi fallback.
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {"pdf": pdf, "png": png, "svg": svg}


def save_long_shap_table(
    df: pd.DataFrame,
    X_scaled_df: pd.DataFrame,
    explanation: shap.Explanation,
    feature_results: pd.DataFrame,
    output_path: Path,
    group_col: str,
) -> None:
    dataset_to_run = dict(zip(feature_results["Dataset_Column"], feature_results["Feature_Name"]))
    roi_col = "INFO_NameOfRoi" if "INFO_NameOfRoi" in df.columns else None
    rows: List[Dict[str, Any]] = []

    for i, idx in enumerate(X_scaled_df.index):
        base = {
            "Row_Index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "PatientID": df.loc[idx, group_col] if group_col in df.columns else None,
            "ROI": df.loc[idx, roi_col] if roi_col else None,
        }
        for j, dataset_col in enumerate(X_scaled_df.columns):
            rows.append({
                **base,
                "Feature_Name": dataset_to_run[dataset_col],
                "Standardized_Feature_Value": float(X_scaled_df.iloc[i, j]),
                "SHAP_Value": float(explanation.values[i, j]),
            })

    pd.DataFrame(rows).to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the three-panel feature-stability + SHAP figure for ADC--Post1--Pre / Linear SVM."
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="ADC--Post1--Pre triple CSV used by the selected run.")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Selected no-SMOTE EN->mRMR run directory containing nested-CV artifacts.")
    p.add_argument("--output-dir", default=Path("Figs"), type=Path,
                   help="Output directory (default: Figs).")
    p.add_argument("--label-col", default="Label")
    p.add_argument("--group-col", default="PatientID")
    p.add_argument("--model", default=MODEL_NAME,
                   help=f"Nested-CV model key (default: {MODEL_NAME}).")
    p.add_argument(
        "--shap-output", choices=["probability", "decision"], default="probability",
        help=("SHAP output to explain. Default probability matches the current Methods text; "
              "decision is exact/faster but requires matching Methods wording."),
    )
    p.add_argument(
        "--allow-reference-mismatch",
        action="store_true",
        help="Do not abort if stability/performance values do not match the frozen ADC--Post1--Pre reference run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print("=" * 88)
    print("BEST-MODEL FEATURE STABILITY + SHAP")
    print("ADC--Post1--Pre | EN -> mRMR | Auto-k | no SMOTE | Linear SVM")
    print("=" * 88)
    print(f"Dataset : {dataset}")
    print(f"Run dir : {run_dir}")
    print(f"Outputs : {output_dir}")

    stability_long, nested_cv_results, config = load_run_artifacts(run_dir)
    recurrence, n_folds, avg_features_per_fold = recurrence_table(stability_long)
    oof = compute_oof_metrics(nested_cv_results, args.model)

    validate_reference_run(
        recurrence, n_folds, avg_features_per_fold, oof, config,
        allow_mismatch=args.allow_reference_mismatch,
    )

    recurrent = recurrence[recurrence["Selection_Count"] >= MIN_RECURRENT_FOLDS].copy()
    if recurrent.empty:
        raise RuntimeError("No recurrent features were selected in at least two outer folds.")

    print("\nReference-run checks passed.")
    print(f"  Outer folds                  : {n_folds}")
    print(f"  Average selected/fold        : {avg_features_per_fold:.1f}")
    print(f"  Unique selected features     : {len(recurrence)}")
    print(f"  Max recurrence               : {int(recurrence['Selection_Count'].max())}/5")
    print(f"  Recurrent features (>=2/5)   : {len(recurrent)}")
    print(f"  Linear-SVM mean AUC          : {oof['mean_auc']:.3f}")
    print(f"  Linear-SVM pooled AUC        : {oof['pooled_auc']:.3f}")

    df = load_dataset(dataset, args.label_col, args.group_col)
    y = pd.to_numeric(df[args.label_col], errors="raise").astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, found: {sorted(np.unique(y).tolist())}")

    fitted_columns, feature_mapping = align_recurrent_features(df, recurrent)

    model, imputer, scaler, X_scaled_df, aggregated_params = fit_interpretation_model(
        df=df,
        feature_columns=fitted_columns,
        y=y,
        nested_cv_results=nested_cv_results,
        model_name=args.model,
    )

    print("\nFull-cohort interpretation refit:")
    print(f"  Recurrent feature count      : {X_scaled_df.shape[1]}")
    print(f"  Aggregated SVM C (median)    : {float(aggregated_params['C']):.6g}")
    print("  SMOTE                        : No")
    print(f"  SHAP output                  : {args.shap_output}")

    explanation = compute_shap(model, X_scaled_df, output_mode=args.shap_output)

    feature_results = build_feature_results(
        recurrence=recurrence,
        recurrent_original_names=recurrent["Feature_Name"].tolist(),
        feature_mapping=feature_mapping,
        X_scaled_df=X_scaled_df,
        explanation=explanation,
    )

    # Keep run recurrence order available, but save table ranked by SHAP for interpretation.
    feature_results = feature_results.sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = output_dir / "best_model_recurrent_feature_stability_and_shap.csv"
    feature_results.to_csv(feature_csv, index=False)

    long_csv = output_dir / "best_model_shap_values_long.csv"
    save_long_shap_table(
        df=df,
        X_scaled_df=X_scaled_df,
        explanation=explanation,
        feature_results=feature_results,
        output_path=long_csv,
        group_col=args.group_col,
    )

    figure_paths = make_figure(
        feature_results=feature_results,
        X_scaled_df=X_scaled_df,
        explanation=explanation,
        output_dir=output_dir,
        shap_output_mode=args.shap_output,
    )

    metadata = {
        "pipeline": {
            "source_configuration": "ADC--Post1--Pre",
            "feature_selection": "Elastic Net -> mRMR",
            "feature_count": "Auto-k in nested CV",
            "smote": False,
            "classifier": "Linear SVM",
        },
        "nested_cv_oof_metrics": oof,
        "stability": {
            "n_outer_folds": n_folds,
            "average_features_per_fold": avg_features_per_fold,
            "n_unique_features_across_folds": int(len(recurrence)),
            "max_selection_count": int(recurrence["Selection_Count"].max()),
            "min_recurrent_folds_for_xai": MIN_RECURRENT_FOLDS,
            "n_recurrent_features": int(len(feature_results)),
        },
        "interpretation_model": {
            "fit_scope": "full cohort, interpretation only",
            "feature_space": "features selected in at least 2 of 5 outer folds",
            "imputation": "full-cohort median",
            "standardization": "full-cohort StandardScaler",
            "hyperparameter_aggregation": "median across outer-fold best_params_per_fold",
            "aggregated_params": aggregated_params,
            "smote": False,
            "shap_explained_output": (
                "Linear SVM class-1 predict_proba" if args.shap_output == "probability"
                else "Linear SVM decision_function"
            ),
            "shap_sign_interpretation": "positive values move model output toward class 1/malignant",
            "performance_note": "Full-cohort refit performance is not used or reported as validation performance.",
        },
        "input": {"dataset": dataset, "run_dir": run_dir},
        "outputs": {
            "figure": figure_paths,
            "feature_table": feature_csv,
            "shap_values_long": long_csv,
        },
    }

    metadata_path = output_dir / "best_model_xai_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(metadata), f, ensure_ascii=False, indent=2)

    print("\nTop recurrent features by mean |SHAP|:")
    show_cols = [
        "Feature_Name", "Selection_Count", "Selection_Frequency",
        "Source", "Feature_Family", "MeanAbsSHAP", "SHAP_Direction"
    ]
    print(feature_results[show_cols].head(15).to_string(index=False))

    print("\nSaved:")
    print(f"  PDF : {figure_paths['pdf']}")
    print(f"  PNG : {figure_paths['png']}")
    print(f"  SVG : {figure_paths['svg']}")
    print(f"  CSV : {feature_csv}")
    print(f"  CSV : {long_csv}")
    print(f"  JSON: {metadata_path}")
    print("\nIMPORTANT: report only the nested-CV/OOF performance values; the full-cohort refit is for interpretation only.")


if __name__ == "__main__":
    main()
