# Machine Learning-Based Classification of Benign and Malignant Breast Lesions Using Radiomic Features from MRI Data (T1 DCE, T2W, and ADC Scans)

---

## Overview

This project focuses on developing machine learning models for classifying breast lesions as **benign** or **malignant** using radiomic features extracted from breast MRI scans.

Breast cancer diagnosis frequently depends on radiological evaluation of MRI images to determine whether a biopsy is necessary. Radiologists assess suspicious lesions and assign **BI-RADS (Breast Imaging-Reporting and Data System)** scores, which guide clinical decision-making.

The goal of this study is to leverage radiomic features derived from multiple MRI modalities to train supervised machine learning models capable of improving lesion classification accuracy and supporting radiological diagnosis.

--- 

## Objectives

* Extract and analyze radiomic features from breast MRI scans.
* Trained supervised machine learning models for lesion classification.
* Evaluate the influence of MRI modality and segmentation variability.
* Investigate model interpretability using explainable AI techniques.
* Assess diagnostic performance using comprehensive statistical metrics.

---

## Dataset and MRI Modalities

MRI scans are collected from patients with breast lesions.

### Included MRI Modalities

#### 1. T1-weighted Dynamic Contrast Enhanced (T1 DCE)

* Captures tumor vascularity
* Shows contrast uptake patterns

#### 2. T2-weighted Imaging (T2W)

* Represents tissue water content
* Highlights edema and fluid accumulation

#### 3. Apparent Diffusion Coefficient (ADC)

* Derived from diffusion-weighted imaging
* Reflects tissue cellularity and diffusion restriction

### Clinical Information

Available metadata may include:

* BI-RADS scores
* Patient clinical information (if relevant)

---

## Methodology

## 1. Data Acquisition

Data consists of:

* Breast MRI scans
* Clinical labels
* Radiologist annotations
* Segmentation masks

Dataset size is expected to be **fewer than 200 samples**, making careful validation essential.

---

## 2. Lesion Segmentation

Two experienced radiologists independently segment breast lesions.

This enables:

* Measurement of **intra-observer variability**
* Measurement of **inter-observer variability**
* Assessment of segmentation reliability

Each lesion, therefore, has **two segmentation masks**.

---

## 3. Radiomic Feature Extraction

Radiomic features are extracted from pre-registered MRI scans across all modalities.

Feature categories may include:

* First-order intensity statistics
* Shape descriptors
* Texture features
* Higher-order transformed features

Radiomic features are extracted separately using both segmentation masks.

---

## 4. Labeling Strategy

Lesions are labeled according to BI-RADS scores.

### Benign

* BI-RADS 1
* BI-RADS 2
* BI-RADS 3

### Malignant

* BI-RADS 4
* BI-RADS 5

This creates a binary classification problem:

[
y \in {Benign,\ Malignant}
]

---

## 5. Machine Learning Pipeline

Supervised machine learning models will classify lesions using extracted radiomic features.

### Individual Dataset Analysis

For each MRI modality:

* T1 DCE
* T2W
* ADC

Two segmentation masks are available.

Thus, there are:

[
3 \text{ scan types} \times 2 \text{ masks} = 6 \text{ datasets}
]

Models will be trained separately on each dataset to evaluate segmentation variability.

---

## Feature Fusion Experiments

Additional experiments combine radiomic features across modalities.

### Pairwise Concatenation

* T1 + T2
* T1 + ADC
* T2 + ADC

### Triple Concatenation

* T1 + T2 + ADC

These experiments assess whether multimodal fusion improves performance.

---

## Candidate Machine Learning Models

Potential models include:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost
* LightGBM
* AdaBoost
* k-Nearest Neighbors
* Multi-Layer Perceptron (MLP)

Model selection will be based on cross-validation performance.

---

## Explainability Analysis

After training, explainability methods will be used to interpret predictions.

### SHAP (SHapley Additive exPlanations)

Used to:

* Estimate feature importance
* Explain local and global model behavior

### LIME (Local Interpretable Model-Agnostic Explanations)

Used to:

* Explain individual predictions
* Understand decision boundaries

Additional visualizations:

* Feature importance plots
* SHAP summary plots
* RADAR plots

---

## Validation Strategy

Since the dataset is small (<200 samples), robust validation is necessary.

Possible validation strategies include:

### Nested Cross-Validation

Recommended because it:

* Reduces overfitting
* Provides reliable generalization estimates
* Separates hyperparameter tuning from evaluation

### Alternative Strategies

* Leave-One-Out Cross Validation (LOOCV)
* Stratified K-Fold Cross Validation

A nested CV is considered the preferred approach.

---

## Performance Metrics

For each model and dataset, the following metrics will be reported.

### Classification Metrics

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1-score

### Diagnostic Metrics

* ROC-AUC
* Positive Predictive Value (PPV)
* Negative Predictive Value (NPV)

### Agreement Metrics

* Cohen’s Kappa

### Statistical Evaluation

* p-value correlation analysis
* Benjamini–Hochberg False Discovery Rate (FDR) correction

### Calibration

* Calibration curve
* Reliability assessment

### Visual Outputs

* Confusion matrix
* ROC curves
* Calibration plots
* Feature importance plots

---

## Project Outputs

Each experiment will save:

* Trained model
* Hyperparameters
* Predictions
* Performance metrics
* Explainability outputs
* Statistical reports
* Visualization figures

---

## Team Composition

### Physicians (2)

Responsibilities:

* Data collection
* Patient clinical information management

### Radiologists (2)

Responsibilities:

* Independent lesion segmentation
* MRI annotation
* Clinical interpretation

### PhD Student

Responsibilities:

* Data handling
* Radiomic feature extraction
* Manuscript preparation

### Bachelor’s / MSc Student

Responsibilities:

* Machine learning pipeline development
* Model training and evaluation

### Supervisors (3 Assistant Professors)

Responsibilities:

* Project oversight
* Scientific guidance
* Research supervision

---

## Expected Outcomes

The project is expected to produce:

* Reliable segmentation masks with quantified observer variability
* Comprehensive radiomic datasets for all MRI modalities
* High-performance ML models for breast lesion classification
* Explainable predictions identifying important radiomic biomarkers
* Complete statistical and diagnostic performance evaluation

---

## Expected Impact

This research aims to support radiologists by providing AI-assisted decision support for breast cancer diagnosis. Accurate lesion classification may:

* Reduce unnecessary biopsies
* Improve early cancer detection
* Increase diagnostic consistency
* Enable more personalized clinical decision-making

---

## Repository Structure

```bash
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train.py
│   ├── evaluation.py
│   └── explainability.py
│
├── models/
├── outputs/
├── figures/
│
└── README.md
```

---

## License

This project is intended for academic and research purposes.
