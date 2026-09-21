<div align="center">

# 🧬 Patient-Aware Explainable Machine Learning for Breast MRI Radiomics

### Single- and multiparametric **MRI radiomics** with **patient-aware nested cross-validation**<br>for preoperative classification of benign and malignant breast lesions

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.13.9-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="Machine Learning" src="https://img.shields.io/badge/ML-10%20classifiers-F7931E?style=flat-square">
<img alt="Validation" src="https://img.shields.io/badge/validation-patient--aware%20nested%20CV-2D6A2D?style=flat-square">
<img alt="MRI Sources" src="https://img.shields.io/badge/MRI%20sources-5-6C8EBF?style=flat-square">
<img alt="Configurations" src="https://img.shields.io/badge/source%20configurations-31-7B61FF?style=flat-square">
<img alt="Explainability" src="https://img.shields.io/badge/XAI-SHAP-8A5A00?style=flat-square">
<img alt="Status" src="https://img.shields.io/badge/status-research%20code-555555?style=flat-square">
</p>

<sub>Computational code and analysis accompanying the manuscript</sub>

</div>

---

## 📋 At a glance

| | |
|---|---|
| 🎯 **Task** | Binary classification of benign vs. malignant/suspicious breast lesions |
| 🏷️ **Reference labels** | BI-RADS 1–3 → benign / low suspicion; BI-RADS 4–5 → malignant / high suspicion |
| 🧲 **MRI inputs** | ADC · DCE-Pre · DCE-Post1 · DCE-Post2 · T2W |
| 🔀 **Source configurations** | **31** total: 5 single + 10 pairwise + 10 triple + 5 quadruple + 1 all-five |
| 🧬 **Radiomic features** | 144 features per MRI source |
| 🧠 **Classifiers** | 10 supervised machine-learning models |
| 🔬 **Feature selection** | Correlation → mRMR · Elastic Net → mRMR · mRMR → Elastic Net |
| 🎛️ **Feature count** | Automatically optimized within the inner CV loop |
| ⚖️ **Class balancing** | SMOTE vs. no-SMOTE |
| 🎲 **Validation** | 5-fold outer + 3-fold inner **patient-aware nested CV** |
| 👁️ **Interpretability** | Feature-selection stability + SHAP |
| 🏆 **Best pipeline** | ADC + Post1 + Pre · Elastic Net → mRMR · Auto-k · no SMOTE · Linear SVM |
| 📈 **Best mean ROC-AUC** | **0.836 ± 0.088** |
| 📊 **Best pooled ROC-AUC** | **0.799** |

> [!IMPORTANT]
> The study is explicitly **patient-aware**: all lesion ROIs from the same patient remain in the same cross-validation partition.  
> Preprocessing, feature selection, SMOTE, and model optimization are restricted to training data to reduce patient-level information leakage.

---

## 📖 Study

This repository accompanies the manuscript:

> **Patient-Aware Explainable Machine Learning for Preoperative Classification of Benign and Malignant Breast Lesions Using Single- and Multiparametric MRI Radiomics**

The study evaluates whether radiomic features derived from different breast MRI sequences provide complementary information for lesion classification and whether combining more MRI sources necessarily improves predictive performance.

The manuscript reports a retrospective cohort of **49 patients** at the study level. In the computational results section, the analyzed radiomic datasets are reported as representing **48 unique patients** and **117 lesion ROIs**. This README preserves that distinction rather than treating the two counts as interchangeable.

---

## 🗂️ Repository layout

The computational experiments are organized by the number of MRI feature sources combined.

| Path | Stage | Purpose |
|---|:---:|---|
| 📦 `Dataset/` | 0 | Pre-extracted radiomic CSV files |
| 1️⃣ `Single/` | 1 | Five single-source experiments |
| 2️⃣ `Pairwise/` | 2 | All 10 two-source combinations |
| 3️⃣ `Triple/` | 3 | All 10 three-source combinations |
| 4️⃣ `Quadruple/` | 4 | All 5 four-source combinations |
| 5️⃣ `AllFive/` | 5 | Complete five-source feature fusion |
| 📊 `Figs/` | — | Publication and diagnostic figures |
| 📄 `README.md` | — | Repository overview and execution guide |

Representative analysis and batch-runner notebooks include:

```text
Single.ipynb
Batch_Run_On_Single_batch.ipynb

Pairwise.ipynb
Batch_Run_All_10_Pairwise.ipynb

Triple/
Quadruple/
AllFive/
```

The same core machine-learning workflow is applied across fusion levels; the main experimental variable is the MRI feature-source combination.

---

## 🧲 MRI protocol and radiomic sources

The study uses five MRI-derived radiomic inputs:

| Source | Description |
|---|---|
| **ADC** | Apparent diffusion coefficient map derived from DWI |
| **DCE-Pre** | Pre-contrast T1-weighted DCE MRI |
| **DCE-Post1** | First post-contrast DCE phase |
| **DCE-Post2** | Second post-contrast DCE phase |
| **T2W** | T2-weighted STIR MRI |

The MRI examinations were acquired on a **1.5 T Siemens MRI system**.

DWI was acquired using b-values of:

```text
50 and 800 s/mm²
```

ADC maps were generated from DWI and used as the diffusion-sensitive radiomic input. DWI itself was not treated as an independent radiomic source.

The DCE acquisition contained one pre-contrast phase followed by six post-contrast phases. The radiomics analysis retained:

```text
Pre
Post1
Post2
```

together with ADC and T2W.

---

## 🩻 Lesion segmentation

Lesions were manually delineated by an experienced breast radiologist using:

```text
ITK-SNAP 2.2.2
```

Three-dimensional lesion ROIs were generated on a selected post-contrast DCE-T1W reference image and exported in NIfTI format.

Because exported masks did not always preserve the original MRI spatial geometry, a custom alignment procedure implemented with **SimpleITK** was used to:

- correct orientation differences;
- match mask geometry to the reference MRI volume;
- transfer image spacing;
- transfer origin;
- transfer direction information.

Corrected masks were visually checked before radiomic analysis.

---

## 🧪 Image preprocessing

MRI preprocessing was performed using:

```text
LIFEx V25.06.1
```

The manuscript describes the following preprocessing settings:

| Step | Setting |
|---|---|
| Intensity normalization | Z-score normalization |
| Spatial resampling | 1 × 1 × 1 mm³ isotropic voxels |
| Gray-level discretization | Absolute discretization |
| Number of gray levels | 64 |
| Texture dimensionality | 3D |
| Neighbour distance | 1 voxel |
| ROI handling | Largest connected component retained |
| ROI union | Disabled |

These preprocessing steps were applied before radiomic feature extraction.

---

## 🧬 Radiomic feature extraction

A total of **144 radiomic features** were extracted independently from each MRI source using LIFEx.

The feature set included:

- first-order intensity features;
- histogram-based descriptors;
- three-dimensional shape features;
- texture features.

Texture families included:

```text
GLCM
GLRLM
GLZLM / GLSZM
NGLDM
```

The manuscript states that feature definitions were consistent with **IBSI** recommendations.

---

## 🔀 Feature-source combinations

All non-empty combinations of the five MRI feature sources were evaluated:

```text
5 single-source
+ 10 pairwise
+ 10 triple-source
+ 5 quadruple-source
+ 1 all-five
= 31 source configurations
```

Feature fusion was performed at the radiomic-feature level by combining matched lesion ROIs across MRI sources.

The complete five-source representation contained:

```text
720 radiomic predictors
```

before feature selection.

---

## 🔄 Machine-learning pipeline

```text
 MRI acquisition
        │
        ▼
 lesion segmentation
        │
        ▼
 image preprocessing
        │
        ▼
 radiomic feature extraction
        │
        ▼
 5 source-specific radiomic tables
 ADC · Pre · Post1 · Post2 · T2W
        │
        ▼
 31 single- and multi-source configurations
        │
        ▼
 patient-aware outer CV — 5 folds
        │
        ▼
 outer-training patients
        │
        ├── median imputation
        ├── variance filtering
        ├── feature selection
        ├── automatic feature-count optimization
        ├── standardization
        ├── optional SMOTE
        └── Optuna hyperparameter optimization
        │
        ▼
 final classifier fitted on outer-training data
        │
        ▼
 untouched outer-test patients
        │
        ├── fold-level ROC-AUC
        ├── pooled out-of-fold predictions
        ├── PR-AUC and threshold metrics
        ├── calibration / Brier score
        └── patient-level bootstrap confidence intervals
        │
        ▼
 feature-selection stability + SHAP interpretation
```

---

## 🔒 Patient-aware nested cross-validation

The lesion ROI is the prediction unit, while the **patient** is the grouping unit.

Some patients contributed more than one ROI. Therefore, allowing ROIs from one patient to appear in both training and test folds could lead to patient-level leakage.

The workflow uses:

```text
Outer CV: 5 patient-aware folds
Inner CV: 3 patient-aware folds
```

The outer loop estimates generalization performance on unseen patients.

The inner loop is used for:

- feature-selection optimization;
- automatic feature-count selection;
- class-balancing evaluation;
- hyperparameter optimization.

All data-dependent processing is restricted to the relevant training partition.

---

## 🧬 Feature selection

Three sequential feature-selection strategies are evaluated.

| Strategy | Description |
|---|---|
| **Correlation → mRMR** | Correlation filtering followed by minimum redundancy maximum relevance |
| **Elastic Net → mRMR** | Embedded Elastic-Net reduction followed by mRMR |
| **mRMR → Elastic Net** | mRMR reduction followed by Elastic-Net ranking |

### Correlation filtering

Predictors are considered redundant when:

```text
|Pearson r| > 0.90
```

Within correlated groups, features are prioritized using direction-independent univariate ROC-AUC.

### mRMR

Minimum Redundancy Maximum Relevance selects features with:

- high relevance to the target;
- low redundancy with already selected features.

Feature relevance is estimated using mutual information.

### Elastic Net

Elastic-Net regularized logistic regression combines:

```text
L1 regularization
+
L2 regularization
```

Selected features are ranked using the absolute magnitude of their learned coefficients.

---

## 🎛️ Automatic feature-number optimization

The final number of retained radiomic features is optimized within the inner cross-validation loop.

Candidate feature subsets are compared using a lightweight logistic-regression model and inner-fold ROC-AUC.

When multiple candidate feature counts show similar performance, the smaller subset is preferred to limit model complexity and overfitting risk.

For the best ADC–Post1–Pre pipeline, the manuscript reports an average of:

```text
9.6 selected features per outer fold
≈ 10 features
```

---

## ⚖️ SMOTE

Class imbalance correction is not applied automatically.

Instead, each modelling strategy is evaluated:

```text
without SMOTE
with SMOTE
```

When enabled, SMOTE is applied:

```text
after feature selection
after standardization
only to training samples
```

SMOTE is never applied to outer-test observations.

The manuscript reports that the effect of SMOTE on mean ROC-AUC was heterogeneous across source configurations:

```text
ΔROC-AUC range: −0.013 to +0.025
```

SMOTE was beneficial for some configurations but the highest-performing overall model was obtained **without SMOTE**.

---

## 🤖 Machine-learning classifiers

Ten supervised classifiers are evaluated.

| Family | Classifiers |
|---|---|
| Linear | Logistic Regression · Linear SVM |
| Kernel-based | RBF SVM |
| Tree ensembles | Random Forest · Extra Trees · Gradient Boosting · XGBoost · LightGBM |
| Distance-based | K-Nearest Neighbors |
| Probabilistic | Gaussian Naive Bayes |

This model set spans linear, nonlinear kernel, ensemble-tree, distance-based, and probabilistic learning approaches.

---

## 🔧 Hyperparameter optimization

Classifier-specific hyperparameters are optimized inside the inner patient-aware cross-validation loop using:

```text
Optuna
Tree-structured Parzen Estimator (TPE)
```

Optimization is performed without access to outer-test observations.

Examples of optimized parameters described in the manuscript include:

| Classifier | Examples |
|---|---|
| Logistic Regression | `C`, `l1_ratio` |
| Linear SVM | `C` |
| RBF SVM | `C`, `γ` |
| Random Forest | number of trees, depth, leaf size, feature sampling |
| Extra Trees | number of trees, depth, leaf size |
| Gradient Boosting | number of estimators, depth, learning rate |
| KNN | neighbors, weighting, distance metric |
| Gaussian NB | variance smoothing |
| XGBoost | estimators, depth, learning rate, subsampling, regularization |
| LightGBM | estimators, depth, leaves, learning rate, subsampling, regularization |

---

## 📊 Performance evaluation

The primary ranking criterion is:

```text
mean ROC-AUC across the five patient-aware outer folds
```

Additional evaluation metrics include:

- PR-AUC;
- accuracy;
- balanced accuracy;
- sensitivity;
- specificity;
- precision;
- F1-score;
- Cohen's κ;
- Brier score;
- calibration.

Binary predictions use a predefined threshold of:

```text
0.5
```

Predictions from all five outer-test folds are combined into pooled **out-of-fold predictions** for overall evaluation.

---

## 📐 Uncertainty estimation

Uncertainty is estimated using **patient-level bootstrap resampling**.

Patients, rather than individual ROIs, are sampled with replacement so that lesions from the same patient remain clustered.

The manuscript uses:

```text
B = 2,000 bootstrap resamples
```

and percentile-based:

```text
95% confidence intervals
```

---

## 🏆 Main results

The highest-ranked pipeline across all 31 source configurations was:

| Component | Selected setting |
|---|---|
| MRI sources | **ADC + Post1 + Pre** |
| Feature selection | **Elastic Net → mRMR** |
| Feature count | **Automatic optimization** |
| Mean selected features | **9.6 ≈ 10** |
| SMOTE | **No** |
| Classifier | **Linear SVM** |
| Mean outer-fold ROC-AUC | **0.836 ± 0.088** |
| Pooled ROC-AUC | **0.799** |
| Mean PR-AUC | **0.879** |
| Pooled PR-AUC | **0.832** |
| Pooled balanced accuracy | **0.713** |
| Brier score | **0.183** |

At the predefined threshold of 0.5, the pooled confusion matrix contained:

```text
True positives:  47
True negatives:  34
False positives: 16
False negatives: 16
```

---

## 📈 Best configuration at each fusion level

| Fusion level | Best source configuration | Feature selection | SMOTE | Classifier | Mean ROC-AUC |
|---|---|---|:---:|---|---:|
| Single | Post2 | EN → mRMR | No | RBF SVM | 0.790 ± 0.127 |
| Pairwise | ADC + T2W | EN → mRMR | No | RBF SVM | 0.799 ± 0.084 |
| Triple | **ADC + Post1 + Pre** | **EN → mRMR** | **No** | **Linear SVM** | **0.836 ± 0.088** |
| Quadruple | ADC + Post2 + Pre + T2W | EN → mRMR | Yes | RBF SVM | 0.802 ± 0.096 |
| All-five | ADC + Pre + Post1 + Post2 + T2W | EN → mRMR | No | RBF SVM | 0.747 ± 0.117 |

A central result of the study is that **adding more MRI sources did not produce a monotonic improvement in classification performance**.

The complete five-source model performed below the selected ADC–Post1–Pre triple-source configuration.

---

## 🧬 Feature-selection comparison

Across the 31 source configurations:

```text
Elastic Net → mRMR : 20 configurations
mRMR → Elastic Net : 11 configurations
Correlation → mRMR : 0 configurations
```

Elastic Net followed by mRMR was therefore the most frequently highest-ranked feature-selection strategy.

---

## 🤖 Classifier comparison

Number of source configurations in which each classifier achieved the highest rank:

| Classifier | Wins |
|---|---:|
| **RBF SVM** | **11** |
| Logistic Regression | 7 |
| Linear SVM | 6 |
| Gaussian Naive Bayes | 3 |
| KNN | 2 |
| XGBoost | 1 |
| LightGBM | 1 |
| Random Forest | 0 |
| Extra Trees | 0 |
| Gradient Boosting | 0 |

RBF SVM was the most frequently selected classifier across source configurations, although the **best overall individual pipeline** used a Linear SVM.

---

## 👁️ Feature stability and SHAP

The optimal ADC–Post1–Pre model was analyzed for feature-selection stability and model interpretability.

Across the five outer folds:

```text
28 unique radiomic features
were selected at least once
```

and:

```text
13 features
were selected in at least two outer folds
```

The manuscript reports that model behavior was driven mainly by:

- ADC-derived first-order intensity features;
- Post1-DCE intensity features;
- Post1-DCE texture features;
- Post1-DCE morphological features.

ADC histogram median-related features showed particularly strong influence.

The interpretation workflow includes:

- SHAP summary plots;
- global mean absolute SHAP importance;
- feature-selection recurrence analysis.

> [!NOTE]
> SHAP values are interpreted as explanations of **model behavior and feature contribution**, not as evidence of direct biological causality.

---

## 📉 Calibration and decision analysis

The best Linear SVM model was additionally evaluated using:

- probability calibration;
- Brier score;
- decision curve analysis.

The manuscript reports a Brier score of:

```text
0.183
```

for the highest-ranked model.

Calibration and decision-curve analyses are used as complementary assessments beyond ROC-AUC.

---

## 💡 Main finding

The main methodological conclusion is:

> **Selective integration of complementary MRI radiomic information is more effective than simply increasing the number of feature sources.**

The strongest model used a selected triple-source combination:

```text
ADC
+
Post1 DCE
+
Pre DCE
```

rather than the complete five-source feature representation.

This result highlights the importance of:

- patient-aware validation;
- dimensionality reduction;
- feature-selection strategy;
- careful class-balancing evaluation;
- model-specific optimization;
- explainability analysis.

---

## ⚙️ Environment

The machine-learning analysis was implemented in:

```text
Python 3.13.9
```

The computational workflow uses Python-based libraries for:

- numerical processing;
- tabular data handling;
- cross-validation;
- feature selection;
- SMOTE;
- hyperparameter optimization;
- machine-learning classifiers;
- SHAP analysis;
- figure generation.

A representative environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
```

and the main computational dependencies installed with:

```bash
python -m pip install \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  imbalanced-learn \
  optuna \
  xgboost \
  lightgbm \
  shap \
  matplotlib \
  joblib \
  papermill \
  nbformat \
  jupyter
```

For exact numerical reproducibility, use the package versions from the environment used for the manuscript experiments when available.

---

## ▶️ Running the computational experiments

### Single-source analysis

```text
Single.ipynb
Batch_Run_On_Single_batch.ipynb
```

A modelling notebook receives a source-specific radiomic CSV and runs the full patient-aware machine-learning pipeline.

Example parameters:

```python
INPUT_FILE = "path/to/ADC_M1.csv"
DATASET_TAG = "ADC"
TARGET_COLUMN = "Label"
```

### Pairwise analysis

```text
Pairwise.ipynb
Batch_Run_All_10_Pairwise.ipynb
```

The pairwise batch workflow evaluates all ten two-source combinations.

### Higher-order feature fusion

The same experimental logic is used for:

```text
Triple/
Quadruple/
AllFive/
```

The source combination changes while the core validation and modelling strategy remains consistent.

### Parameterized notebook execution

Batch runners can execute notebooks programmatically using Papermill.

Example:

```bash
papermill Single.ipynb Single_ADC_executed.ipynb \
  -p INPUT_FILE "path/to/ADC_M1.csv" \
  -p DATASET_TAG "ADC" \
  -p TARGET_COLUMN "Label"
```

---

## 📤 Generated outputs

<details>
<summary><b>Show representative analysis artifacts</b></summary>

```text
<experiment-output>/
├── dataset validation / QC summaries
├── multi-source matching logs
├── patient-level fold assignments
├── selected-feature tables
├── automatic feature-count results
├── hyperparameter optimization outputs
├── outer-fold predictions
├── pooled out-of-fold predictions
├── ROC-AUC and PR-AUC summaries
├── threshold-dependent metrics
├── confusion matrices
├── calibration summaries
├── Brier scores
├── bootstrap confidence intervals
├── feature-selection recurrence tables
├── SHAP outputs
├── executed notebooks
├── publication figures
└── publication-oriented CSV tables
```

</details>

---

## 🔁 Reproducibility and leakage control

The workflow includes several safeguards intended to reduce optimistic bias in a small, high-dimensional medical-imaging dataset:

- 🔒 patient-level grouping throughout validation;
- 🔁 nested separation of model selection and final evaluation;
- 🧹 training-derived median imputation;
- 🧮 training-derived variance filtering;
- 🧬 training-only feature selection;
- 🎛️ feature-count optimization within model-development data;
- 📏 training-derived standardization;
- ⚖️ training-only SMOTE;
- 🔧 inner-loop hyperparameter optimization;
- 📈 outer-fold out-of-sample predictions;
- 👥 patient-cluster bootstrap uncertainty estimation;
- 👁️ feature-stability and SHAP analyses.

---

## ⚠️ Limitations

The manuscript identifies several limitations:

- retrospective study design;
- single-center cohort;
- relatively limited sample size;
- sensitivity of radiomic features to acquisition and reconstruction parameters;
- dependence on preprocessing and segmentation;
- no integration of clinical variables, molecular subtype information, or BI-RADS descriptors into the predictive feature set;
- need for external multicenter validation.

The reported results should therefore be interpreted as **internal patient-aware validation**, not evidence of established clinical deployment performance.

---

## 🧪 Research and clinical use

This repository is intended for **research and reproducibility**.

It is not intended for direct clinical diagnosis or autonomous clinical decision-making.

The manuscript states that further validation in larger multicenter cohorts is required before clinical translation.

The underlying patient data are not distributed through this repository. According to the manuscript, datasets may be available from the corresponding author upon reasonable request, subject to applicable ethical and institutional requirements.

---

## 📖 Citation

> Yazdani E, Entezari M, Farzanehgan Z, Nazari H, Mirzaee E, Bagherpour Z, Fadavi P,  
> Hosseini Toudeshki S, Abdollahzadeh H, Safari M, Kheradpisheh SR, Beigi M.  
> *Patient-Aware Explainable Machine Learning for Preoperative Classification of Benign and Malignant Breast Lesions Using Single- and Multiparametric MRI Radiomics.*  
> 2026. Manuscript.

```bibtex
@article{yazdani2026patientaware,
  title   = {Patient-Aware Explainable Machine Learning for Preoperative Classification of Benign and Malignant Breast Lesions Using Single- and Multiparametric MRI Radiomics},
  author  = {Yazdani, Elmira and Entezari, Mahla and Farzanehgan, Zahra and Nazari, Hengameh and Mirzaee, Elahe and Bagherpour, Zahra and Fadavi, Pedram and Hosseini Toudeshki, Saeed and Abdollahzadeh, Hasan and Safari, Mojtaba and Kheradpisheh, Saeed Reza and Beigi, Manijeh},
  year    = {2026},
  note    = {Manuscript}
}
```

---

<div align="center">

**Breast MRI · Radiomics · Patient-Aware Nested CV · Feature Selection · SMOTE · Optuna · SVM · SHAP · Explainable AI**

</div>
