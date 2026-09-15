#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate diagnostic performance metrics from pooled
out-of-fold predictions.

Designed for nested patient-aware cross-validation outputs.

Metrics:
- Sensitivity
- Specificity
- PPV
- NPV
- F1-score
- Accuracy
- Balanced Accuracy
- Cohen's Kappa
- ROC-AUC
- PR-AUC
- Confusion matrix

Input:
prepared_oof_predictions_*.csv

Output:
diagnostic_metrics.csv
"""


import argparse
import os
import pandas as pd
import numpy as np


from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)


# --------------------------------------------------
# Automatically detect columns
# --------------------------------------------------

def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Cannot find column. Available columns:\n{df.columns.tolist()}"
    )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

def calculate_metrics(
        y_true,
        y_score,
        threshold=0.5
):

    y_pred = (
        np.asarray(y_score) >= threshold
    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()


    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )


    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )


    results = {

        "N_samples":
            len(y_true),

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,


        "Sensitivity":
            sensitivity,


        "Specificity":
            specificity,


        "PPV":
            ppv,


        "NPV":
            npv,


        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),


        "Balanced_Accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),


        "F1_score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),


        "Cohen_Kappa":
            cohen_kappa_score(
                y_true,
                y_pred
            ),


        "Pooled_AUC":
            roc_auc_score(
                y_true,
                y_score
            ),


        "Pooled_PR_AUC":
            average_precision_score(
                y_true,
                y_score
            ),

        "Threshold":
            threshold
    }


    return results



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():


    parser = argparse.ArgumentParser(
        description=
        "Calculate OOF diagnostic metrics"
    )


    parser.add_argument(
        "--input",
        required=True,
        help="OOF prediction CSV file"
    )


    parser.add_argument(
        "--output",
        default="diagnostic_metrics.csv"
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )


    args = parser.parse_args()



    print("="*70)
    print("OOF DIAGNOSTIC METRIC CALCULATION")
    print("="*70)


    print(
        "Input:",
        args.input
    )


    df = pd.read_csv(
        args.input
    )


    print("\nColumns:")
    print(df.columns.tolist())


    # true label
    y_true_col = find_column(
        df,
        [
            "y_true",
            "true_label",
            "label",
            "target",
            "y"
        ]
    )


    # probability / decision score
    y_score_col = find_column(
        df,
        [
            "y_score",
            "probability",
            "prob",
            "prediction_probability",
            "y_prob",
            "decision_score"
        ]
    )


    print("\nUsing columns:")
    print(
        "True:",
        y_true_col
    )

    print(
        "Score:",
        y_score_col
    )


    y_true = df[y_true_col].astype(int)

    y_score = df[y_score_col].astype(float)



    metrics = calculate_metrics(
        y_true,
        y_score,
        threshold=args.threshold
    )


    output_df = pd.DataFrame(
        [metrics]
    )


    output_df.to_csv(
        args.output,
        index=False
    )


    print("\nResults:")
    print(output_df.T)


    print(
        "\nSaved:",
        args.output
    )


if __name__ == "__main__":
    main()