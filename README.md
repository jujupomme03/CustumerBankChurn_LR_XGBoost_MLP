# Bank Customer Churn Prediction

## Overview

This project tackles a binary classification problem: predicting whether a bank customer will leave (`Exited = 1`) based on their demographic and account characteristics. It is built as an **educational project**, with the intent of comparing three model families of increasing complexity — Logistic Regression, XGBoost, and a Multi-Layer Perceptron — and drawing honest conclusions about when complexity helps and when it does not.

The central argument running through the project is this: **a more complex model is not automatically a better model**. The right choice depends on the nature of the dataset, the business objective, and the evaluation metric. This project demonstrates that concretely, on a real dataset, with real trade-offs.

---

## Dataset

**Source:** [Churn Modelling Dataset — Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling)

| Property | Value |
|---|---|
| Rows | 10,000 customers |
| Features | 13 original (10 retained after cleaning) |
| Target | `Exited` — binary (0: stayed, 1: churned) |
| Class distribution | ~79.6% non-churn / ~20.4% churn |

### Features used

| Feature | Type | Description |
|---|---|---|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | Country (France, Germany, Spain) |
| Gender | Binary | 0 = Female, 1 = Male |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years as a customer |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products held |
| HasCrCard | Binary | Has a credit card |
| IsActiveMember | Binary | Active member status |
| EstimatedSalary | Numeric | Estimated annual salary |

---

## Project Structure

```
.
├── 1_BankCustomersChurn_DataPrep.ipynb      # Data cleaning, EDA, feature engineering, export
├── 2_BankCustomersChurn_Run_Models.ipynb    # Model training, evaluation, comparison
├── BankCustomersSet_XGBoost_MLP.csv         # Processed dataset for XGBoost and MLP
├── BankCustomersSet_LR.csv                  # Processed dataset for Logistic Regression
└── README.md
```

---

## Methodology

### Notebook 1 — Data Preparation

**Cleaning:** No missing values or duplicates. `RowNumber`, `CustomerId`, and `Surname` are dropped as non-predictive identifiers.

**Preprocessing:** `Gender` is label-encoded (binary). `Geography` is one-hot encoded with `drop_first=True` to avoid multicollinearity.

**Exploratory Data Analysis:**
- `Age` is the strongest individual predictor of churn (older customers churn more).
- `IsActiveMember` is negatively correlated with churn.
- `Balance` is bimodal: a large group has zero balance, another group clusters around 100k–150k.
- `Geography_Germany` shows a disproportionate churn rate relative to its share of customers.
- Most features have low individual correlation with the target, confirming that churn is driven by feature combinations rather than a single dominant variable.

**Feature Engineering:**

| Feature | Logic |
|---|---|
| `Balance_Is_Zero` | Flags customers with no active balance |
| `Balance_Salary_Ratio` | Relative financial exposure |
| `IsActive_x_NumProducts` | Inactive customers with multiple products are high-risk |
| `Tenure_Per_Age` | Contextualises tenure relative to customer age |
| `Age_Squared` | Captures non-linearity in age (LR only) |
| `Germany_x_Balance` | Interaction term for German customers with high balances (LR only) |

A second multicollinearity check is run after feature engineering. For Logistic Regression specifically, five highly correlated original features are removed to keep the coefficient interpretation valid.

---

### Notebook 2 — Model Training and Evaluation

#### Class imbalance management

| Model | Strategy |
|---|---|
| Logistic Regression | SMOTE on training set |
| XGBoost | `scale_pos_weight` parameter (native) |
| MLP (PyTorch) | SMOTE on training set |

SMOTE is applied strictly **after** the train/test split to prevent any information from the test set leaking into the synthetic minority samples.

#### Evaluation metrics

Standard accuracy is not used. The project focuses on:

- **PR-AUC** (primary): measures quality on the minority class (churners). A random model scores ~0.20 (the base rate); higher is better.
- **F2-Score**: weights recall twice as much as precision, reflecting the asymmetric cost of missing a churner.
- **ROC-AUC**: reported for completeness, but noted as potentially misleading on imbalanced data.

#### Threshold optimisation

All three models use a **custom decision threshold** rather than the default 0.5. The threshold that maximises the F2-score on the test set is selected for each model individually, acknowledging that in a churn context, false negatives (missed churners) are more costly than false positives.

---

## Models

### Logistic Regression

**Tuning:** `GridSearchCV` over `C` (regularisation strength) and penalty type (`L1`/`L2`), 5-fold cross-validation, scored on ROC-AUC.

**Strengths in this context:**
- Directly interpretable: coefficients show the direction and magnitude of each feature's influence.
- Stable and fast to train and tune.
- The regularisation curve shows that performance is stable across a wide range of `C` values, meaning the model is robust and not sensitive to the exact hyperparameter choice.

**Weaknesses:**
- Cannot capture non-linear relationships or feature interactions without explicit engineering (hence `Age_Squared`, `Germany_x_Balance`).
- Requires a separate multicollinearity management step that reduces the feature set, limiting information available to the model.

---

### XGBoost

**Tuning:** `RandomizedSearchCV` with 50 iterations over 7 hyperparameters, 5-fold cross-validation, scored on ROC-AUC.

**Strengths in this context:**
- Natively handles class imbalance via `scale_pos_weight` without resampling.
- Captures non-linear relationships and feature interactions automatically, without needing interaction terms or polynomial features.
- The loss curve (log-loss per boosting round) provides a direct view of convergence and generalisation behaviour.
- Consistently produces the best PR-AUC and F2-Score on this dataset.

**Weaknesses:**
- Less interpretable by default (requires SHAP or feature importance analysis to explain individual predictions).
- More hyperparameters to manage; results can be sensitive to the search space definition.

---

### Multi-Layer Perceptron (PyTorch)

**Architecture:** `[input → 128 → 64 → 32 → 1]` with ReLU activations and Dropout (p=0.3) between layers. Output is a raw logit; `BCEWithLogitsLoss` is used as the loss function.

**Optimiser:** Adam with `lr=0.001` and L2 regularisation via `weight_decay=1e-4`.

**Training:** 100 epochs with a 90/10 internal train/validation split. Train loss and validation loss (BCE) are recorded at each epoch and plotted on the same axis for direct comparison.

**Strengths in this context:**
- The explicit training loop exposes convergence behaviour in a way that sklearn's `MLPClassifier` does not: the dual loss curve shows whether the model is overfitting, underfitting, or training stably.
- Dropout and weight decay give fine-grained control over regularisation without relying on a pipeline abstraction.
- Scales well to larger datasets and can be extended with more sophisticated architectures.

**Weaknesses and why this matters here:**
- On a tabular dataset of 10,000 rows with mostly engineered features, the MLP has limited opportunity to express its representational advantage over tree-based methods.
- Training is less stable than XGBoost: results depend on the random initialisation, learning rate, and number of epochs.
- Requires more infrastructure (manual training loop, tensor conversions, scaler management outside the model) for the same or lower performance than XGBoost on this specific dataset.

---

## Key Takeaway: Complexity is Not a Virtue

This project is designed to illustrate a principle that is often overlooked in practice.

XGBoost outperforms the MLP on this dataset despite being a shallower model in terms of representational capacity. The reasons are structural:

1. **Dataset size:** 10,000 rows is insufficient for a neural network to gain a decisive advantage over a well-tuned gradient boosting model. Neural networks typically need orders of magnitude more data to leverage their depth.

2. **Feature type:** The dataset is entirely tabular, with numerical and low-cardinality categorical features. Gradient boosting methods are specifically designed for this setting. Neural networks excel on unstructured data (images, text, audio).

3. **Class imbalance:** XGBoost handles imbalance natively through `scale_pos_weight`, which adjusts the loss function directly. The MLP requires external resampling (SMOTE), which introduces synthetic samples and adds a preprocessing dependency.

4. **Evaluation objective:** The project optimises for recall on the minority class (F2-Score, PR-AUC). XGBoost's probability outputs tend to be well-calibrated on tabular data, which makes threshold optimisation more reliable.

At the same time, Logistic Regression — the simplest model — produces results competitive with the MLP, at a fraction of the computational cost and with full coefficient interpretability. This is a direct consequence of the feature engineering step, which manually encoded the non-linearity (via `Age_Squared`) and interaction effects (via `Germany_x_Balance`) that XGBoost discovers automatically.

The lesson is not that simple models are always better. It is that **model selection should be driven by the data, the objective, and the constraints** — not by the appeal of complexity.

---

## Business Interpretation

A model that achieves ~75–80% recall on churners means that out of every 100 customers about to leave, roughly 75–80 are correctly identified before they do. In a customer base of 10,000 with a 20% churn rate (~2,000 churners), this translates to approximately **1,500–1,600 actionable alerts per period**.

If a targeted retention campaign converts 20% of those flagged customers back into active members, and each retained customer generates €500/year in margin, the model's output represents **€150,000–€160,000 in preserved annual revenue** — from a dataset that required no new data collection, only better modelling.

Precision matters here too: a threshold set too low generates too many false alarms, overloading the retention team and diluting the campaign's impact. The F2-based threshold optimisation explicitly navigates this trade-off.

---

## Limitations and Next Steps

**Explainability:** SHAP values would allow the retention team to understand, for each flagged customer, which features drove the prediction. This is critical for personalising the intervention.

**Threshold monitoring:** The optimal threshold was calibrated on the test set. In production, it should be recalibrated periodically as the customer distribution shifts over time.

**Data drift:** Customer behaviour changes. A monitoring layer (e.g. EvidentlyAI) should track feature distributions and model performance metrics in production.

**Deployment:** The final model (scaler + model weights) can be serialised (`torch.save` for the MLP, `joblib` for XGBoost) and served via a lightweight FastAPI endpoint for real-time or batch scoring.

**Extended feature engineering:** External data such as transaction frequency, recent product usage changes, or customer service interactions could significantly improve recall on the hardest-to-predict churners.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
torch
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost torch
```

---

## Authors

Project developed as part of a Master's programme in Artificial Intelligence.  
Dataset source: [Kaggle — Churn Modelling](https://www.kaggle.com/datasets/shubh0799/churn-modelling)
