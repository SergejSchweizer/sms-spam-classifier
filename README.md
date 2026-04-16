# SMS Spam Classifier

This repository contains an end-to-end SMS spam classification workflow, from exploratory analysis to model training and error analysis.

## Table of Contents

- [Setup Environment](#setup-environment)
- [Windows PowerShell](#windows-powershell)
- [Linux](#linux)
- [Current Implementation](#current-implementation)
- [Dataset Source](#dataset-source)
- [Method Overview](#method-overview)
- [Text Representation](#text-representation)
- [Modeling Approach](#modeling-approach)
- [Models Used in the Notebook](#models-used-in-the-notebook)
- [Hyperparameter Optimization per Model](#hyperparameter-optimization-per-model)
- [Evaluation](#evaluation)
- [Utility Functions](#utility-functions)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Setup Environment

Create and activate a local virtual environment before installing dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

To deactivate:

```powershell
deactivate
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

To deactivate the environment:

```bash
deactivate
```

## Current Implementation

- Reusable text/feature utilities are in `src/utils.py`.
- Modeling strategies and cross-validation trainer are in `src/modeling.py`.
- Training/evaluation pipeline helpers are in `src/training_utils.py`.
- `notebooks/sms_spam_supervised_model_selection.ipynb` contains the main supervised learning workflow: feature engineering, multi-model comparison, Bayesian hyperparameter optimization, stratified k-fold evaluation, and final holdout-test assessment with recall as the primary optimization target.
- `notebooks/sms_spam_semantic_embedding_clustering.ipynb` explores transformer-based semantic embeddings, normalization, dimensionality reduction, and unsupervised clustering analysis with KMeans and DBSCAN.
- The end-to-end modeling notebook uses a clean train/test split for final evaluation, with feature transforms fitted on train data and applied to test data.

## Dataset Source

The original dataset source was not available for direct download, so this project uses the Kaggle mirror:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

Before running the notebook/pipeline, prepare the data file:

1. Download the dataset archive from the Kaggle link above.
2. Unzip the archive.
3. Place `spam.csv` in the `data/raw/` folder (inside the `data` subdirectory), so the expected path is `data/raw/spam.csv`.

## Method Overview

### Text Representation

The feature space combines dense handcrafted features and sparse text features:

- Dense features: message length, word/digit/uppercase counts, punctuation counts, ratio features, and spam-oriented indicator/count features.
- Sparse features: TF-IDF over n-grams (`1` to `3`) with chi-squared feature selection (`SelectKBest`) to keep the most discriminative terms.

The helper `add_selected_tfidf_features(...)` encapsulates TF-IDF fitting + chi-squared selection and returns selected feature columns plus fitted transformers.

### Modeling Approach

The notebook uses a **strategy-based modeling pipeline**:

- a shared train/test split for fair model comparison
- Optuna-based hyperparameter optimization
- stratified cross-validation with spam recall as optimization target
- one common training/evaluation flow across model families (Strategy Pattern in `src/modeling.py`)

Optimization metric detail: Optuna maximizes the **mean cross-validated recall for the spam class**.

For modeling, we intentionally used the **Strategy Design Pattern** because it keeps model-specific logic isolated while sharing one consistent training workflow. This was chosen to:

- make model comparison fair and reproducible (same pipeline, different strategy)
- add or swap models with minimal code changes
- improve maintainability by avoiding duplicated training/evaluation code
- keep experiment code easier to read and extend over time

### Models Used in the Notebook

The notebook compares the following models:

- **Logistic Regression** (`LogisticRegressionStrategy`)
- **Random Forest** (`RandomForestStrategy`)
- **Multinomial Naive Bayes** (`MultinomialNBStrategy`)

Default Logistic Regression settings in the strategy are:

- `solver="liblinear"`
- `class_weight="balanced"`
- `max_iter=2000`

### Hyperparameter Optimization per Model

The following parameters are optimized with Optuna in the strategy classes:

- **Logistic Regression** (`LogisticRegressionStrategy`)
- optimized: `C` (log scale, `1e-3` to `10`)
- fixed during optimization: `solver="liblinear"`, `class_weight="balanced"`, `max_iter=2000`

- **Random Forest** (`RandomForestStrategy`)
- optimized: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight` (`None` or `"balanced"`)
- fixed during optimization: `random_state=42`, `n_jobs=-1`

- **Multinomial Naive Bayes** (`MultinomialNBStrategy`)
- optimized: `alpha` (log scale, `1e-3` to `10`), `fit_prior` (`True`/`False`)

### Evaluation

Evaluation is recall-focused while still reporting a full metric set:

- cross-validated recall during optimization
- holdout test metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC)
- false-negative and false-positive analysis
- coefficient-based feature importance for interpretability

## Utility Functions

`src/utils.py` provides:

- `preprocess(text)` for normalization, tokenization, and stopword removal
- `top_ngram_summary(df, n, top_k)` for class-wise n-gram comparison
- `add_selected_tfidf_features(...)` for TF-IDF + chi-squared feature selection and feature-frame augmentation

`src/training_utils.py` provides:

- `optimize_strategy(...)` for Optuna optimization with CV recall objective
- `train_final_model(...)` for final fit on full training data
- `evaluate_final_model(...)` for metric computation on holdout data
- `analyze_errors(...)` for false-negative/false-positive inspection
- `run_strategy_pipeline(...)` for end-to-end tuning + training + evaluation flow

## Project Structure

```text
Sms_spam_analysis/
|-- data/
|   `-- raw/
|-- notebooks/
|-- src/
|   |-- __init__.py
|   |-- modeling.py
|   |-- training_utils.py
|   `-- utils.py
`-- tests/
```

## Dependencies

Pinned core dependencies are managed in `requirements.txt`, including:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `optuna`
- `ipywidgets`
