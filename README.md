# Breast Cancer Diagnosis: Inference & Classification

**Claire Remolano · April 2026**

Statistical inference and machine learning classification applied to the
Wisconsin Breast Cancer Diagnostic dataset to identify tumor cell nuclei
characteristics associated with malignancy and build a high-performance
diagnostic model.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Setup and Usage](#setup-and-usage)
- [Requirements](#requirements)
- [Limitations and Future Work](#limitations-and-future-work)
- [AI Disclaimer](#ai-disclaimer)

---

## Project Overview

This project has two complementary objectives:

**1. Statistical inference**: Identify which tumor cell nuclei characteristics
robustly predict malignancy, using Firth's logistic regression with 5,000
iteration bootstrap stability analysis to account for quasi-separation and
small-sample bias.

**2. Predictive classification**: Build and compare 13 classification models
optimized for clinical recall (minimizing false negatives), with SHAP-based
explainability for the best model.

The work is designed to support diagnostic decision-making by combining
statistical rigor with predictive performance, contributing to both
interpretability and trust in machine learning applications for medical
diagnosis.

---

## Dataset

**Source**: [UCI ML Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
via `sklearn.datasets.load_breast_cancer`

| Property | Value |
|---|---|
| Samples | 569 tumor cell nuclei |
| Features | 30 numerical (image-derived) |
| Target (original) | 1 = Benign, 0 = Malignant |
| Target (re-labeled) | **1 = Malignant, 0 = Benign** |
| Class distribution | 37.3% malignant, 62.7% benign |
| Missing values | None |

Features are computed from digitized fine-needle aspirate biopsy images and
summarized three ways per nucleus: **mean** (average), **worst** (mean of the
three largest values), and **error** (standard error). Each summarizes
measurements for radius, texture, perimeter, area, smoothness, compactness,
concavity, concave points, symmetry, and fractal dimension.

---

## Repository Structure

```
.
├── bc_inference_classification.ipynb   # Main analysis notebook
├── modeling.py                         # All model classes:
│                                       #   PreprocessingPipeline
│                                       #   FirthLogisticModel, MLELogisticModel
│                                       #   LogisticInferencePipeline, InferenceResult
│                                       #   BootstrapInference
│                                       #   PipelineBuilder, ModelTrainer
│                                       #   CrossValidator, ModelAnalyzer
│                                       #   ModelEvaluator, ModelVisualizer
├── data_processing.py                  # SafeTransformer, DataFrameScaler,
│                                       #   VIFSelector, identify_skewed_features
├── eda.py                              # EDA helper functions and grid plots
├── requirements.txt                    # Pinned dependencies
└── README.md
```

---

## Methodology

### Feature Selection

| Group | Description | Inference | Classification |
|---|---|---|---|
| `worst_*` | Mean of 3 largest nuclei values | ✅ Selected | ✅ Selected |
| `mean_*` (selected) | Average nuclei values | Partial (low-collinearity only) | ✅ Broader set |
| `*_error` | Standard error across nuclei | ❌ Dropped | ❌ Dropped |
| `*_area`, `*_perimeter` | Geometric functions of radius | ❌ Dropped | ❌ Dropped |

The inference model uses a narrower feature set to prioritize coefficient
interpretability. The classification models use a broader set and rely on
pipeline-level VIF filtering and regularization to handle collinearity.

### Preprocessing Pipeline (model-dependent)

```
Log-transform (skew > 1)  →  StandardScaler  →  VIF filter (thresh = 3.0)
```

Applied to: logistic regression variants, SVM, KNN.

Tree-based models (XGBoost, CatBoost, LightGBM, RandomForest, DecisionTree):
no transformation or scaling applied since these algorithms are invariant to
feature transformations and do not require normalized inputs.

### Statistical Inference

| Component | Choice | Rationale |
|---|---|---|
| Estimator | Firth penalized logistic regression | Handles quasi-complete separation and reduces small-sample MLE bias |
| Inference | Likelihood-ratio test (LRT) p-values | More accurate than Wald under separation |
| CIs | Profile-likelihood | Correct coverage under separation and flagged `unstable_ci=True` when optimizer fails |
| Bootstrap | 5,000 resamples with replacement | Empirical stability assessment across the full pipeline |
| Stability metric | Selection frequency (p < 0.05 AND VIF-retained) | Measures joint significance and retained features after collinearity is considered |

### Classification

| Component | Choice |
|---|---|
| Models compared | 13 (tree-based, linear, distance-based) |
| Hyperparameter tuning | RandomizedSearchCV, 20 iterations, PR-AUC scoring (handles class imbalance) |
| Cross-validation | Stratified 5-fold |
| Threshold optimization | Coarse-to-fine F2 grid search per fold |
| Primary selection metric | **F2 score** (weights recall 2× over precision) |
| Explainability | SHAP (TreeExplainer for tree models) |

---

## Key Results

### Inference (Bootstrapped Firth's Logistic Regression)

| Feature | OR per SD | 95% CI | Selection Freq. | Robust |
|---|---|---|---|---|
| worst radius (log) | >> 1000 | Unstable — quasi-separation | 99.8% | ⚠️ Direction only |
| worst texture | 5.70 | 3.13 – 11.54 | 100% | ✅ |
| worst smoothness | 5.09 | 2.32 – 12.66 | 91.2% | ✅ |
| mean smoothness | — | — | ~83% | ✅ Bootstrap only |
| mean symmetry | 2.76 | 1.13 – 6.87 | 43.3% | ❌ |

> ORs are per one standard deviation increase in the standardized
> (log-transformed where applicable) feature, controlling for all others.

**Worst radius** near-perfectly separates malignant from benign tumor cell
nuclei. Firth's penalty shrinks but cannot fully stabilize its coefficient
under quasi-separation; it is flagged as unstable and reported directionally
only.

**Worst texture** and **worst smoothness** are the robust, reportable
predictors with stable sign, consistent magnitude, and high selection frequency
across 5,000 bootstrap resamples.

**Mean smoothness** was identified as important in bootstrap stability despite
not surviving the initial VIF filter, likely reflecting shared variance with
correlated features. Its consistent selection indicates that surface
irregularity contributes to malignancy beyond the features directly included in
the model.

### Classification Model Results

| Model | F2 | Recall | Precision | Brier Score | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.987** | **98.4%** | **100%** | **0.0159** | 0.995 |
| Lasso | 0.978 | **98.4%** | 95.6% | 0.021 | 0.997 |
| CatBoost | 0.975 | 96.9% | 100% | 0.020 | **0.998**|
| KNN | 0.972 | **98.4%** | 92.6% | 0.037 | 0.978 |
| Random Forest | 0.966 | 96.9% | 95.4% | 0.025 | 0.997 |


Optimal classification thresholds ranged from 0.20 to 0.38 across top models,
compared to the default of 0.5. Threshold optimization meaningfully improved
recall without sacrificing calibration.

### Model Explainability (SHAP — XGBoost)

- **Worst radius** is the dominant predictor, with a non-linear threshold
  near values of 15–18. Below this range, the model strongly predicts benign cases;
  above it the prediction shifts sharply toward malignancy.
- **Mean concave points** is the second most influential feature, with a clear
  threshold near 0.05 above which malignancy predictions are consistently strong.
- **Worst concavity** acts as a context-dependent modifier where it reinforces
  malignant predictions in combination with high worst radius, but contributes
  weakly in isolation.
- **Symmetry and fractal dimension** contributed minimally, consistent with
  their low bootstrap selection frequency in the inference model.

### Error Analysis

The XGBoost model resulted in **1 false negative** on the test set where a malignant sample
was predicted as benign with high model confidence (predicted probability = 0.004).
SHAP waterfall analysis showed that dominant features pushed the prediction
toward benign while weaker malignancy signals were insufficient to overcome
them. The sample's feature values fall within the transition zone between peaks
of multimodal feature distributions, indicating a decision-boundary case rather
than a data quality issue. This underscores the importance of integrating model
predictions with clinical judgment rather than relying on them in isolation.

---

## Setup and Usage

### Run in Google Colab (recommended)

1. Upload the repository files to Google Drive
2. Open `bc_inference_classification.ipynb` in Colab
3. Update `BASE_PATH` in the drive mount cell to your own drive path
4. Run all cells. The first cell installs dependencies from `requirements.txt`

```python
# Update this path before running
BASE_PATH = Path('/content/drive/MyDrive/your_project_folder')
```

> **Resumability**: `ModelEvaluator` saves fitted pipelines, CV results, and
> thresholds to `BASE_PATH` after each stage. Re-running the notebook in a
> new session automatically skips completed stages and reloads from disk.
> Delete the output directory to force a full re-run from scratch.

### Run locally

```bash
pip install -r requirements.txt
jupyter notebook bc_inference_classification.ipynb
```

### Key notebook calls

```python
# --- Inference ---
inf_model = LogisticInferencePipeline(
    estimator=FirthLogisticModel(),
    intent={'transform': True, 'scale': True, 'vif': True},
    vif_thresh=3.0,
)
inf_model.fit(X_infer, y)

inf_result  = inf_model.get_result()
inf_table   = inf_result.to_table()         # includes unstable_ci flag
display(inf_table)

bootstrap_model = BootstrapInference(base_model=inf_model, n_bootstrap=5000)
bootstrap_model.fit(X_infer, y)

forest_data = bootstrap_model.build_forest_data(inf_table)
inf_result.plot_forest(data=forest_data, ...)

# --- Classification ---
evaluator = ModelEvaluator(
    X_train, y_train, X_test, y_test, cv,
    models, param_grids, model_intent,
    output_dir=BASE_PATH,
)
analyzer   = evaluator.run()          # skips completed stages automatically
visualizer = ModelVisualizer(analyzer)

visualizer.plot_shap_analysis(name='XGBoost')
visualizer.plot_shap_dependence()
visualizer.explain_confident_errors_with_shap()
```

---

## Requirements

Key dependencies. Full list in `requirements.txt`:

```
scikit-learn == 1.6.1
xgboost == 3.1.2
lightgbm == 4.6.0
catboost == 1.2.8
firthmodels == 0.6.0
shap == 0.50.0
statsmodels == 0.14.6
numpy == 2.0.2
pandas == 2.2.2
matplotlib == 3.10.0
seaborn == 0.13.2
```

---

## Limitations and Future Work

**Limitations**

- Small dataset (569 samples); results may not generalize without external
  validation on an independent cohort
- Features are image-derived engineered measurements; deep learning on raw
  biopsy images may capture additional signal not accessible here
- No patient-level clinical covariates (age, family history, prior biopsy
  results) are available
- Quasi-separation in log-transformed worst radius limits the reliability of its
  coefficient point estimate in the inference model, even with Firth penalization

**Future Work**

- External validation on independent patient cohorts
- Deep learning applied directly to raw biopsy image data
- Incorporation of patient demographic and clinical history features
- Decision-curve analysis to better align classification thresholds with
  clinical risk trade-offs and downstream costs of false negatives

---

## AI Disclaimer

AI tools were used to support code development, debugging, and refinement of
explanations throughout this project. All code was reviewed and validated by
the author. Final interpretations and conclusions reflect independent analysis.

---

*Dataset: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
UCI Machine Learning Repository.*
