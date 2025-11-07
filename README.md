Sure — here’s a clean, **plain Markdown** version of your README, written in a professional tone and ready to drop straight into GitHub as `README.md`:

---

# Adult Income Classification – Machine Learning Project 2

**Authors:** William Donnell-Lonon and Emma Ewing
**Dataset:** 1994 U.S. Census Income Database (Adult dataset)
**Goal:** Predict whether an individual earns more than $50K per year using a Multilayer Perceptron (MLP) neural network.

---

## Overview

This project applies Multilayer Perceptrons (MLPs) to predict income levels based on demographic and employment features.
Two models were compared:

1. A **Baseline MLP** trained extensively on a fixed set of hyperparameters.
2. A **Tuned MLP** using GridSearchCV to optimize hyperparameters through cross-validation.

The objective was to determine whether hyperparameter tuning leads to measurable performance improvements and to analyze the trade-offs between training depth, computational cost, and model complexity.

---

## Data

**Source:** Extracted from the 1994 Census database.
**Files:**

* `project_adult.csv` – training dataset
* `project_validation_inputs.csv` – validation dataset

**Features include:**
`age`, `workclass`, `fnlwgt`, `education`, `marital-status`, `occupation`, `relationship`, `race`,
`sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, and `income`.

**Target variable:**
`income` – binary classification (`>50K` vs `≤50K`)

**Data summary:**

* Training set: 26,048 rows, 16 columns
* Validation set: 6,513 rows, 15 columns (income excluded)
* Imbalanced target: 75.9% ≤50K, 24.1% >50K
* Missing values handled via mode imputation for `workclass`, `occupation`, and `native-country`

---

## Preprocessing

Preprocessing was performed through a custom function and pipeline:

* Standardized categorical values (spacing and case)
* Mode imputation for missing categorical values
* One-hot encoding for categorical features
* Income label recoded as binary
* `StandardScaler` applied within the training pipeline

---

## Exploratory Data Analysis (EDA)

* Most individuals are aged between 20–40 years.
* Higher education levels correlate with higher income likelihood.
* Individuals working more than 40 hours per week are significantly more likely to earn >50K.
* The dataset shows notable class imbalance, motivating F1 and ROC-AUC metrics for evaluation.

---

## Model Architecture

### Baseline MLP

Trained with a single configuration emphasizing full convergence.

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    learning_rate_init=0.001,
    batch_size=256,
    max_iter=200,
    tol=1e-4,
    n_iter_no_change=10,
    early_stopping=True,
    random_state=42
)
```

### Tuned MLP (GridSearchCV)

Searched multiple network architectures and learning rates.

```python
param_grid = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64), (96, 48)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__learning_rate_init": [5e-4, 1e-3],
    "mlp__alpha": [1e-4, 1e-3, 1e-2]
}
```

3-fold stratified cross-validation was used, refitting on the F1 metric.

---

## Results

| Metric   | Baseline | Tuned (GridSearchCV) |
| :------- | :------- | :------------------- |
| Accuracy | ~0.86    | ~0.85                |
| F1 Score | ~0.64    | ~0.62                |
| ROC AUC  | ~0.90    | ~0.89                |

**Interpretation:**

* The tuned model performed nearly identically to the baseline.
* The baseline trained longer and more deeply due to stricter convergence settings (`tol=1e-4`, `n_iter_no_change=10`).
* The tuned model explored more hyperparameters but used looser early stopping (`tol=1e-3`, `n_iter_no_change=8`), reducing training depth.

---

## How to Improve

1. **Match training depth:**
   Align parameters like `tol`, `n_iter_no_change`, and `batch_size` between models for fair comparison.

2. **Use stronger cross-validation:**
   Increase folds (e.g., 5-fold or repeated CV) for more stable tuning results.

3. **Optimize search efficiency:**
   Use `RandomizedSearchCV` or `HalvingGridSearchCV` to explore parameter space more efficiently.

4. **Expand parameter tuning:**
   Include parameters such as `batch_size`, `tol`, and `n_iter_no_change` in the search grid.

5. **Threshold calibration:**
   Adjust decision thresholds on validation data to maximize F1 score.

6. **Ensure preprocessing consistency:**
   Keep all transformations inside the pipeline to prevent data leakage.

---

## Ethical Considerations

* The dataset reflects historical socioeconomic biases, which may perpetuate unfair outcomes in real-world applications.
* Models trained on this data could reinforce income, gender, or racial disparities if deployed in hiring or lending contexts.
* MLPs are inherently opaque; model interpretability techniques (e.g., SHAP, LIME, permutation importance) should be used to ensure accountability.
* Economic data from 1994 may not generalize to current societal and labor conditions.

---

## Conclusion

The tuned MLP explored more configurations but trained less deeply, resulting in similar or slightly lower performance than the baseline.
Model performance is influenced not only by hyperparameter choice but also by training consistency and computational constraints.
Future work should balance model complexity, interpretability, and computational efficiency to achieve fair, explainable, and resource-conscious modeling.

---

## Files

| File                            | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| `ML_Project2.9.ipynb`           | Main notebook for preprocessing, model training, and evaluation |
| `ML_Project2.pptx`              | Presentation summarizing project results and conclusions        |
| `project_adult.csv`             | Training dataset                                                |
| `project_validation_inputs.csv` | Validation dataset                                              |

---

Would you like me to add a short “how to run” section (dependencies and steps to execute the notebook)? It would make this perfect for GitHub.
