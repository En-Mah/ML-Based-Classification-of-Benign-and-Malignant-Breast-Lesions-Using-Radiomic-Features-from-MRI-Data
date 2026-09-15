#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ROC and calibration curves for all classifiers
on the same dataset configuration.

Uses pooled out-of-fold predictions.
"""


import os
import glob
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

from sklearn.calibration import calibration_curve



# --------------------------------------------------
# Find columns
# --------------------------------------------------

def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Cannot find column. Available:\n{df.columns}"
    )



# --------------------------------------------------
# Model name cleanup
# --------------------------------------------------

def clean_model_name(name):

    replacements = {

        "LogisticRegression":
            "Logistic Regression",

        "SVM_Linear":
            "Linear SVM",

        "SVM_RBF":
            "RBF SVM",

        "RandomForest":
            "Random Forest",

        "GradientBoosting":
            "Gradient Boosting",

        "XGBoost":
            "XGBoost",

        "LightGBM":
            "LightGBM",

        "KNN":
            "KNN",

        "NaiveBayes":
            "Naive Bayes",

        "GaussianNB":
            "Naive Bayes"
    }


    for key,value in replacements.items():

        if key in name:
            return value


    return name



# --------------------------------------------------
# Load OOF files
# --------------------------------------------------

def load_oof_files(folder):

    files = glob.glob(
        os.path.join(
            folder,
            "**",
            "prepared_oof_predictions*.csv"
        ),
        recursive=True
    )


    if len(files)==0:

        raise FileNotFoundError(
            "No OOF prediction files found"
        )


    print("\nFound files:")

    for f in files:
        print(f)


    return files



# --------------------------------------------------
# ROC plot
# --------------------------------------------------

def plot_roc(models, output):


    plt.figure(figsize=(7,6))


    results=[]


    for model,data in models.items():

        y_true=data["y_true"]

        y_prob=data["y_prob"]


        auc=roc_auc_score(
            y_true,
            y_prob
        )


        results.append(
            (model,auc)
        )


        fpr,tpr,_ = roc_curve(
            y_true,
            y_prob
        )


        plt.plot(
            fpr,
            tpr,
            linewidth=1.5,
            label=f"{model} (AUC={auc:.3f})"
        )


    # identify best model

    best_model=max(
        results,
        key=lambda x:x[1]
    )[0]


    print(
        "\nBest ROC model:",
        best_model
    )


    # redraw best curve thicker

    y_true=models[best_model]["y_true"]

    y_prob=models[best_model]["y_prob"]


    fpr,tpr,_=roc_curve(
        y_true,
        y_prob
    )

    auc=roc_auc_score(
        y_true,
        y_prob
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=3,
        label=f"BEST: {best_model} (AUC={auc:.3f})"
    )


    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        linewidth=1
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curves - ADC Post1 Pre Fusion"
    )


    plt.legend(
        fontsize=8,
        loc="lower right"
    )


    plt.tight_layout()


    plt.savefig(
        output+".png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output+".pdf",
        bbox_inches="tight"
    )


    plt.close()




# --------------------------------------------------
# Calibration plot
# --------------------------------------------------

def plot_calibration(models, output):


    plt.figure(figsize=(7,6))


    for model,data in models.items():

        y_true=data["y_true"]

        y_prob=data["y_prob"]


        prob_true,prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="quantile"
        )


        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1.5,
            label=model
        )


    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        linewidth=1
    )


    plt.xlabel(
        "Mean predicted probability"
    )


    plt.ylabel(
        "Observed probability"
    )


    plt.title(
        "Calibration Curves - ADC Post1 Pre Fusion"
    )


    plt.legend(
        fontsize=8
    )


    plt.tight_layout()


    plt.savefig(
        output+".png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output+".pdf",
        bbox_inches="tight"
    )


    plt.close()



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--input-dir",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        default="Figs"
    )


    args=parser.parse_args()


    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    files=load_oof_files(
        args.input_dir
    )


    models={}


    for f in files:


        df=pd.read_csv(f)


        y_true_col=find_column(
            df,
            [
                "y_true",
                "true_label",
                "label"
            ]
        )


        y_prob_col=find_column(
            df,
            [
                "y_prob",
                "y_score",
                "probability"
            ]
        )


        model=df["Model"].iloc[0] \
            if "Model" in df.columns \
            else os.path.basename(f)


        model=clean_model_name(
            str(model)
        )


        models[model]={

            "y_true":
                df[y_true_col].values,

            "y_prob":
                df[y_prob_col].values
        }



    print(
        "\nModels loaded:"
    )

    for m in models:
        print(m)



    plot_roc(
        models,
        os.path.join(
            args.output_dir,
            "ROC_all_models_ADC_Post1_Pre"
        )
    )


    plot_calibration(
        models,
        os.path.join(
            args.output_dir,
            "Calibration_all_models_ADC_Post1_Pre"
        )
    )


if __name__=="__main__":
    main()