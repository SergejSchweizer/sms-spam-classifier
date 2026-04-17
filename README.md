# SMS Spam Classifier

This project analyzes SMS messages and classifies them as `spam` or `ham` (not spam). It includes both a supervised classification workflow and an embedding-based clustering exploration.

## Table of Contents

- [Project Goal](#project-goal)
- [Project Scope](#project-scope)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How To Run](#how-to-run)
- [Supervised Learning Approach](#supervised-learning-approach)
- [Clustering / Embedding Exploration](#clustering--embedding-exploration)
- [Code Overview](#code-overview)
- [Design Decision](#design-decision)
- [Dependencies](#dependencies)
- [Discussion Points For The Appointment](#discussion-points-for-the-appointment)
- [Possible Short Submission Note](#possible-short-submission-note)

## Project Goal

The main goal is to build and compare approaches for SMS spam detection and to document the reasoning behind the chosen solution.

The project covers:

- exploratory analysis of SMS messages
- text preprocessing and feature engineering
- supervised spam classification
- hyperparameter optimization
- model evaluation and error analysis
- semantic clustering with embeddings

## Project Scope

This repository covers the following areas:

- **Classical Machine Learning & Data Science**
  - word and n-gram analysis in spam vs. ham messages
  - supervised SMS spam classification model
  - clearly documented Python implementation
- **GenAI / LLM**
  - semantic clustering of SMS messages with embeddings
  - external embedding models can be used and discussed separately in the appointment

## Dataset

The project uses the **SMS Spam Collection Dataset** via the Kaggle mirror:

<https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download>

After downloading:

1. Extract the archive.
2. Place `spam.csv` in `data/raw/`.
3. The expected path is `data/raw/spam.csv`.

## Repository Structure

```text
sms-spam-classifier/
|-- data/
|   `-- raw/
|-- notebooks/
|   |-- sms_spam_supervised_model_selection.ipynb
|   `-- sms_spam_semantic_embedding_clustering.ipynb
|-- src/
|   |-- __init__.py
|   |-- modeling.py
|   |-- training_utils.py
|   `-- utils.py
|-- tests/
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
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

## How To Run

After setting up the environment and placing the dataset in `data/raw/spam.csv`, start Jupyter and open the notebooks:

```powershell
jupyter notebook
```

Recommended order:

1. `notebooks/sms_spam_supervised_model_selection.ipynb`
2. `notebooks/sms_spam_semantic_embedding_clustering.ipynb`

## Supervised Learning Approach

The supervised workflow focuses on identifying whether an SMS is spam or ham.

### Feature Representation

The model combines:

- handcrafted text features such as message length, uppercase count, digit count, punctuation count, and spam-related keyword indicators
- TF-IDF features with n-grams from `1` to `3`
- chi-squared feature selection to keep the most informative terms

### Models Compared

The following models are implemented through a strategy-based design:

- Logistic Regression
- Random Forest
- Multinomial Naive Bayes

### Optimization and Evaluation

The training pipeline uses:

- stratified cross-validation
- Optuna for hyperparameter optimization
- **recall for the spam class** as the main optimization target

Reported evaluation includes:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC-AUC (when available)
- false positive / false negative analysis

## Clustering / Embedding Exploration

The second notebook explores semantic similarity between messages using embeddings.

This part includes:

- text cleanup for embedding generation
- semantic vector representations of SMS messages
- clustering experiments with KMeans and DBSCAN
- cluster quality analysis using metrics such as silhouette score

This section is useful because it shows how message meaning can be represented beyond simple keyword counting.

## Code Overview

- `src/utils.py`
  - preprocessing helpers
  - n-gram summaries
  - TF-IDF feature generation
  - clustering helper functions
- `src/modeling.py`
  - strategy pattern for different model families
  - reusable cross-validation trainer
- `src/training_utils.py`
  - hyperparameter optimization
  - final model training
  - evaluation and error analysis

## Design Decision

The project uses the **Strategy Pattern** for the supervised models.

Why this was chosen:

- each model has its own parameter space and construction logic
- the training and evaluation workflow stays the same across models
- new model families can be added with little code duplication
- model comparison becomes cleaner and easier to explain

## Dependencies

Main libraries used in this project:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `optuna`
- `ipywidgets`
- `sentence-transformers`

The exact versions are listed in `requirements.txt` and `pyproject.toml`.

## Discussion Points For The Appointment

These are good points to explain during the review:

1. Why spam recall was prioritized over plain accuracy.
2. Why TF-IDF + engineered features were combined.
3. Why Logistic Regression, Random Forest, and Naive Bayes were selected for comparison.
4. Why the Strategy Pattern was used in the code design.
5. What semantic embeddings add compared to classical bag-of-words features.
6. What the main false positives and false negatives reveal about the dataset.

## Possible Short Submission Note

If you need a short note for the appointment or upload text, you can use:

> This repository contains an SMS spam classification project with exploratory text analysis, a supervised machine learning pipeline with model comparison and hyperparameter tuning, and an embedding-based clustering exploration for semantic grouping of messages.
