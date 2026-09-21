<div align="center">

# 🧬 Patient-Aware Breast MRI Radiomics

### Explainable **patient-aware machine learning** with **single- and multi-source MRI radiomics**<br>for benign vs. malignant breast lesion classification

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.13.9-3776AB?style=flat-square&logo=python&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white">
<img alt="Validation" src="https://img.shields.io/badge/validation-patient--aware%20nested%20CV-2D6A2D?style=flat-square">
<img alt="Configurations" src="https://img.shields.io/badge/MRI%20configurations-31-6C8EBF?style=flat-square">
<img alt="Explainability" src="https://img.shields.io/badge/XAI-SHAP-8A5A00?style=flat-square">
<img alt="Status" src="https://img.shields.io/badge/status-research%20code-555555?style=flat-square">
</p>

<sub>Computational code accompanying the manuscript</sub>

</div>

---

## 📋 At a glance

| | |
|---|---|
| 🎯 **Task** | Binary classification of benign vs. malignant breast lesions from radiomic features |
| 🧲 **MRI sources** | ADC · Pre · Post1 · Post2 · T2 |
| 🔀 **Feature-source configurations** | **31** total: 5 single + 10 pairwise + 10 triple + 5 quadruple + 1 all-five |
| 🧠 **Classifiers** | 10 supervised ML models |
| 🧬 **Feature selection** | Correlation → mRMR · Elastic Net → mRMR · mRMR → Elastic Net |
| 🎛️ **Feature count** | Auto-k selected inside training data |
| ⚖️ **Class balancing** | SMOTE vs. no-SMOTE |
| 🎲 **Validation** | 5-fold outer + up to 3-fold inner **patient-aware nested CV** |
| 🔧 **Optimization** | Optuna with TPE sampling |
| 👁️ **Interpretability** | Feature-selection stability + SHAP |
| 🏆 **Best pipeline** | ADC + Post1 + Pre · EN → mRMR · Auto-k · no SMOTE · Linear SVM |
| 📈 **Best mean ROC-AUC** | **0.836 ± 0.088** |

> [!IMPORTANT]
> The machine-learning analysis starts from **pre-extracted tabular radiomic features**.  
> MRI acquisition, lesion segmentation, image preprocessing, and radiomic feature extraction are described in the associated manuscript and are not part of the downstream model-fitting notebooks.

---

## 🗂️ Repository layout

| Path | Stage | Purpose |
|---|:---:|---|
| 📦 `Dataset/` | 0 | Source radiomic CSV files used by the modelling pipeline |
| 1️⃣ `Single/` | 1 | Single-source analysis for ADC, Pre, Post1, Post2, and T2 |
| 2️⃣ `Pairwise/` | 2 | All 10 two-source feature-fusion experiments |
| 3️⃣ `Triple/` | 3 | All 10 three-source feature-fusion experiments |
| 4️⃣ `Quadruple/` | 4 | All 5 four-source feature-fusion experiments |
| 5️⃣ `AllFive/` | 5 | Complete five-source feature concatenation |
| 📊 `Figs/` | — | Publication and diagnostic figures |
| 📄 `README.md` | — | Repository overview and execution guide |

Representative modelling and batch-runner notebooks include:

```text
Single.ipynb
Batch_Run_On_Single_batch.ipynb

Pairwise.ipynb
Batch_Run_All_10_Pairwise.ipynb

Triple/
Quadruple/
AllFive/
```

Each experiment group follows the same modelling logic; only the input feature-source combination changes.

---

## 🧲 MRI feature sources

Five MRI-derived radiomic feature sources are evaluated:

| Source | Description |
|---|---|
| **ADC** | Apparent diffusion coefficient map |
| **Pre** | Pre-contrast DCE T1-weighted MRI |
| **Post1** | First post-contrast DCE T1-weighted MRI |
| **Post2** | Second post-contrast DCE T1-weighted MRI |
| **T2** | T2-weighted MRI |

All non-empty combinations are tested:

```text
5 single-source
+ 10 pairwise
+ 10 triple-source
+ 5 quadruple-source
+ 1 all-five
= 31 source configurations
```

For every source configuration:

```text
3 feature-selection strategies
× 2 SMOTE conditions
× 10 classifiers
= 60 primary pipelines
```

Across the complete study:

```text
31 × 60 = 1,860 primary machine-learning pipelines
```

---

## 🔄 Pipeline

```text
 five pre-extracted MRI radiomic feature tables
 ADC · Pre · Post1 · Post2 · T2
        │
        │  match ROIs and build source combinations
        ▼
 31 validated source configurations
        │
        │  patient-aware 5-fold outer CV
        ▼
 outer-training patients
        │
        ├── median imputation
        ├── variance filtering
        ├── feature selection
        ├── Auto-k feature-count selection
        ├── standardisation
        ├── optional SMOTE
        └── Optuna hyperparameter optimisation
        │
        ▼
 final model fitted on outer-training patients
        │
        ▼
 untouched outer-test patients
        │
        ├── fold-level metrics
        ├── out-of-fold predictions
        ├── pooled ROC / PR evaluation
        ├── calibration and Brier score
        └── patient-cluster bootstrap CIs
        │
        ▼
 feature-selection stability + SHAP interpretation
```

Two properties of the validation design are central to how the reported results should be interpreted:

- 🔒 **All ROIs from the same patient stay in the same fold.** A patient cannot contribute one ROI to training and another ROI to validation or testing.
- ✅ **All data-dependent operations are training-only.** Imputation, variance filtering, feature selection, standardisation, SMOTE, and hyperparameter optimisation are fitted without access to the corresponding outer-test patients.

---

## 📦 Expected input format

The modelling notebooks consume CSV files containing one row per ROI.

| Field | Role |
|---|---|
| `INFO_PatientName` / `PatientID` | Patient grouping variable used for patient-aware splitting |
| `INFO_NameOfRoi` | ROI identifier used for matching observations across MRI sources |
| `Label` | Binary target: `0 = benign`, `1 = malignant` |
| Numerical columns | Candidate radiomic predictors |

For multi-source experiments, matched ROIs are joined **column-wise**. Feature names retain a source prefix so the origin of every predictor remains traceable.

Examples:

```text
ADC__GLCM_Correlation
Post1__GLCM_Correlation
T2__GLCM_Correlation
```

<details>
<summary><b>Show a representative project tree</b></summary>

```text
Dataset/
└── Breast-Data/
    └── Mask1/
        ├── ADC_M1.csv
        ├── Pre_M1.csv
        ├── Post1_M1.csv
        ├── Post2_M1.csv
        └── T2_M1.csv

Single/
├── Single.ipynb
└── Batch_Run_On_Single_batch.ipynb

Pairwise/
├── Pairwise.ipynb
└── Batch_Run_All_10_Pairwise.ipynb

Triple/
├── generated triple-source datasets
├── executed notebooks
├── nested_cv_outputs_all_triple_concat_batch_smote_compare/
└── summary tables/

Quadruple/
├── generated quadruple-source datasets
├── batch notebooks
└── outputs/

AllFive/
├── complete five-source dataset
├── modelling notebooks
└── outputs/

Figs/
└── publication figures
```

</details>

---

## 🧬 Feature selection

Three sequential feature-selection strategies are compared.

| Strategy | Description |
|---|---|
| **Correlation → mRMR** | Removes strongly correlated variables, then selects informative low-redundancy predictors |
| **Elastic Net → mRMR** | Supervised regularised screening followed by mRMR refinement |
| **mRMR → Elastic Net** | mRMR candidate reduction followed by Elastic-Net ranking |

### Correlation filtering

Two predictors are treated as strongly correlated when:

```text
|r| > 0.90
```

When several predictors are strongly correlated, the workflow prioritises them using direction-independent univariate ROC-AUC.

### Auto-k

The number of final selected features is chosen inside the model-development data rather than fixed globally.

Candidate values are:

```text
k ∈ {4, 6, 8, 10, 12, 14, 16, 18}
```

A lightweight logistic-regression model compares the candidate values using patient-aware inner-fold ROC-AUC. If two values tie, the smaller feature set is preferred.

---

## ⚖️ SMOTE

Class balancing is treated as an experimental factor rather than a mandatory preprocessing step.

Each feature-selection strategy is tested:

```text
without SMOTE
with SMOTE
```

When enabled, SMOTE is applied only after the selected features have been standardised:

```text
imputation
→ variance filtering
→ feature selection
→ standardisation
→ optional SMOTE
→ classifier
```

Validation and outer-test observations are never synthetically oversampled.

---

## 🤖 Machine-learning models

Ten supervised classifiers are evaluated on the same selected feature representation.

| Family | Classifiers |
|---|---|
| Linear | Logistic Regression · Linear SVM |
| Kernel | RBF SVM |
| Tree ensembles | Random Forest · Extra Trees · Gradient Boosting · XGBoost · LightGBM |
| Distance-based | K-Nearest Neighbours |
| Probabilistic | Gaussian Naive Bayes |

Classifier-specific hyperparameters are selected independently inside the patient-aware inner cross-validation loop.

---

## 🎛️ Shared modelling configuration

| Setting | Value |
|---|---|
| Prediction unit | ROI / lesion |
| Grouping unit | Patient |
| Outer cross-validation | 5 patient-aware folds |
| Inner cross-validation | Up to 3 patient-aware folds |
| CV splitter | `StratifiedGroupKFold` |
| Random seed | 100 |
| Primary ranking metric | Mean outer-fold ROC-AUC |
| Feature-count candidates | 4, 6, 8, 10, 12, 14, 16, 18 |
| Feature-selection strategies | 3 |
| Class-balancing conditions | SMOTE / no SMOTE |
| Classifiers | 10 |
| Optuna sampler | TPE |
| Optuna trials | 10 per classifier / outer-fold study |
| Bootstrap resamples | 2,000 patient-cluster resamples |
| Classification threshold | 0.5 for threshold-dependent metrics |

---

## 🔧 Hyperparameter optimisation

Classifier hyperparameters are tuned with Optuna using:

```python
optuna.samplers.TPESampler(seed=100)
```

A separate optimization study is created for each:

```text
source configuration
× feature-selection strategy
× SMOTE condition
× classifier
× outer fold
```

Each study evaluates 10 candidate hyperparameter settings.

The corresponding outer-test fold is not used during tuning.

---

## 📊 Performance evaluation

The primary ranking criterion is:

```text
mean ROC-AUC across the five patient-aware outer folds
```

Additional metrics include:

- ROC-AUC
- PR-AUC
- accuracy
- balanced accuracy
- sensitivity
- specificity
- precision
- F1-score
- Cohen's κ
- Brier score
- calibration

Predictions from the outer-test folds are concatenated into an **out-of-fold (OOF)** prediction vector for pooled evaluation.

Uncertainty is estimated with **patient-cluster bootstrap resampling** so that all ROIs from one patient remain grouped during resampling.

---

## 🏆 Main results

The highest-ranked pipeline in the complete experiment was:

| Component | Selected setting |
|---|---|
| MRI sources | **ADC + Post1 + Pre** |
| Feature selection | **Elastic Net → mRMR** |
| Feature count | **Auto-k** |
| Mean selected features | **9.6 ≈ 10** |
| SMOTE | **No** |
| Classifier | **Linear SVM** |
| Mean outer-fold ROC-AUC | **0.836 ± 0.088** |
| Pooled ROC-AUC | **0.799** |
| Pooled PR-AUC | **0.832** |
| Pooled balanced accuracy | **0.713** |
| Pooled F1-score | **0.746** |

Best configuration at each fusion level:

| Fusion level | Best source configuration | Mean ROC-AUC |
|---|---|---:|
| Single | Post2 | 0.790 |
| Pairwise | ADC + T2 | 0.799 |
| Triple | **ADC + Post1 + Pre** | **0.836** |
| Quadruple | ADC + Post2 + Pre + T2 | 0.802 |
| All-five | ADC + Pre + Post1 + Post2 + T2 | 0.747 |

The central computational finding is that **adding more MRI sources did not consistently improve classification performance**. The strongest result was obtained from a selected triple-source representation rather than complete five-source concatenation.

---

## 👁️ Feature stability and SHAP

Feature-selection stability and SHAP answer two different questions:

- **Selection stability:** how consistently is a feature retained across patient-aware outer folds?
- **SHAP:** how strongly does a recurrent feature influence the fitted interpretation model?

Selection frequency across five folds is recorded as:

```text
5 / 5 folds → 1.0
4 / 5 folds → 0.8
3 / 5 folds → 0.6
2 / 5 folds → 0.4
1 / 5 folds → 0.2
```

SHAP outputs include:

- global mean absolute SHAP importance;
- summary dot plots;
- direction and magnitude of feature contributions.

> [!NOTE]
> SHAP values explain the behaviour of the fitted model. They should not be interpreted as evidence of biological causality.

---

## ⚙️ Environment

The manuscript workflow was developed with **Python 3.13.9** and Jupyter notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main packages used by the analysis:

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
  statsmodels \
  papermill \
  nbformat \
  jupyter
```

For exact numerical reproducibility, use the pinned environment associated with the final manuscript runs when available.

> The workflow uses `seed = 100` wherever the underlying library supports an explicit random seed.

---

## ▶️ Running the experiments

The project uses interactive modelling notebooks and batch-runner notebooks.

### Single-source analysis

```text
Single.ipynb
Batch_Run_On_Single_batch.ipynb
```

Typical notebook parameters include:

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

The batch runner evaluates all ten two-source combinations.

### Triple, quadruple, and all-five analysis

The same execution pattern is used for:

```text
Triple/
Quadruple/
AllFive/
```

### Papermill example

Parameterized notebooks can be executed with Papermill:

```bash
papermill Single.ipynb Single_ADC_executed.ipynb \
  -p INPUT_FILE "path/to/ADC_M1.csv" \
  -p DATASET_TAG "ADC" \
  -p TARGET_COLUMN "Label" \
  -p IS_BATCH_RUN true \
  -p RUN_PUBLICATION_PLOTS false
```

---

## 📤 Output of an experiment

<details>
<summary><b>Show representative generated artifacts</b></summary>

```text
<experiment-output>/
├── dataset / merge / QC logs
├── patient-fold assignments
├── selected-feature tables
├── Auto-k results
├── Optuna parameters and inner-CV scores
├── fold-level predictions
├── pooled out-of-fold predictions
├── ROC-AUC / PR-AUC metrics
├── threshold-dependent metrics
├── calibration / Brier summaries
├── patient-cluster bootstrap confidence intervals
├── feature-selection stability tables
├── SHAP outputs
├── executed notebooks
├── publication figures
└── publication-oriented CSV summaries
```

</details>

The saved OOF predictions allow pooled performance analyses to be reproduced without evaluating a model on observations that were used to fit that model.

---

## 🔁 Reproducibility and leakage control

The workflow is designed to reduce optimistic bias in a small, high-dimensional radiomics setting.

Key safeguards include:

- 🔒 patient-level grouping in every validation split;
- 🔁 nested separation of model development and outer-test evaluation;
- 🧹 training-only median imputation and variance filtering;
- 🧬 training-only feature selection and Auto-k;
- 📏 training-only standardisation;
- ⚖️ training-only SMOTE;
- 🎛️ inner-loop hyperparameter optimisation;
- 🎲 fixed random seed where supported;
- 📈 pooled evaluation from OOF predictions;
- 👥 patient-cluster bootstrap uncertainty estimates;
- 💾 saved fold assignments, selected features, parameters, predictions, logs, and executed notebooks.

---

## 🧪 Research use

This repository is intended for **research and reproducibility**.

It is not intended for direct clinical diagnosis or clinical decision-making.

The reported performance is based on internal patient-aware validation. Larger independent and multicentre validation is required before clinical translation.

Patient-level imaging data or derived datasets should only be distributed when permitted by the relevant ethics approval, institutional policy, and data-use agreements.

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

<div align="center">
<sub>Breast MRI · Radiomics · Patient-Aware Nested CV · Feature Selection · SMOTE · Optuna · SVM · SHAP</sub>
</div>
