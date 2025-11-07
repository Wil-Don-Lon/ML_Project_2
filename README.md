# Multi-Layer Perceptron Income Classification

## Project Overview

This project implements a Multi-Layer Perceptron (MLP) neural network to predict whether an individual earns more than $50,000 annually based on demographic and employment data from the Adult Census Income dataset. The implementation includes comprehensive preprocessing, hyperparameter tuning via GridSearchCV, and performance evaluation across multiple metrics.

## Dataset

The project uses two primary datasets:
- `project_adult.csv`: Training data with income labels
- `project_validation_inputs.csv`: Unlabeled validation data for final predictions

### Features
The dataset includes both categorical and numerical features:
- **Numerical**: age, education_num, hours_per_week, capital_gain, capital_loss, etc.
- **Categorical**: workclass, education, marital_status, occupation, relationship, race, sex, native_country

### Target Variable
- Income level: Binary classification (<=50K coded as 0, >50K coded as 1)
- Class distribution is imbalanced, with approximately 76% earning <=50K

## Methodology

### Data Preprocessing

1. **Data Cleaning**
   - Standardized column names (lowercase, underscore-separated)
   - Replaced missing value indicators ("?") with NaN
   - Normalized categorical values

2. **Feature Engineering**
   - Categorical features: Imputation (most frequent) and one-hot encoding
   - Numerical features: Imputation (median) and standardization
   - Pipeline integration for consistent transformations

3. **Train-Test Split**
   - 80/20 stratified split to maintain class distribution
   - Random state fixed for reproducibility

### Model Architecture

**Baseline MLP**
- Single hidden layer configuration
- Fixed hyperparameters with early stopping
- Max iterations: 150
- Batch size: 256
- Learning rate: 0.001

**Tuned MLP (GridSearchCV)**
- 5-fold stratified cross-validation
- Hyperparameter search space:
  - Hidden layer sizes: (64,), (128,), (128, 64), (96, 48)
  - Activation functions: relu, tanh
  - Alpha (L2 regularization): 1e-4, 1e-3, 1e-2
  - Initial learning rate: 5e-4, 1e-3
- Multiple scoring metrics: accuracy, F1, ROC AUC
- Refit strategy: Optimized for F1 score

### Evaluation Metrics

- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed breakdown of prediction outcomes

## Results

### Model Performance

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Baseline MLP | 0.847 | 0.660 | 0.909 |
| Tuned MLP | 0.847 | 0.649 | 0.908 |

### Key Findings

1. Both models achieve approximately 85% accuracy on the holdout set
2. The baseline MLP slightly outperforms the tuned model in F1 score (0.660 vs 0.649) and ROC AUC (0.909 vs 0.908)
3. The tuned model exhibits a higher false negative rate (516 vs 480), missing more high-income individuals
4. Both models correctly predict the majority of <=50K cases but struggle with the minority >50K class (approximately 41% false negative rate for tuned, 38% for baseline)
5. The baseline model achieves better recall for the positive class (0.62 vs 0.59), likely because the tuned model required more extensive training to fully converge with its optimized architecture
6. Limited training iterations (max_iter=200) may have prevented the tuned model from reaching its full potential, suggesting that extended training could improve performance
7. Feature importance analysis reveals education level, age, and hours worked per week as primary predictors

### Visualizations

The notebook includes:
- Training loss curves showing convergence behavior
- Confusion matrix demonstrating classification performance
- ROC curves illustrating trade-offs between true and false positive rates
- Feature importance rankings via permutation importance
- Class distribution comparisons between true and predicted labels

## Reproducibility

All random operations use `random_state=42` for consistent results across runs. The preprocessing pipeline is cached using joblib to accelerate repeated experiments.

## Output

Final predictions are saved to `Group_11_MLP_PredictedOutputs.csv` with the following format:
- Value 1: Income >50K
- Value -1: Income <=50K

## Suggestions for Improvement

### Model Enhancement

1. **Class Imbalance Handling**
   - Implement SMOTE (Synthetic Minority Over-sampling Technique) or class weighting
   - Adjust decision threshold based on business requirements
   - Use stratified sampling with more sophisticated balancing techniques

2. **Feature Engineering**
   - Create interaction features (e.g., age x education_num)
   - Bin continuous variables (age groups, income brackets)
   - Engineer domain-specific features (e.g., education-to-age ratio)
   - Consider polynomial features for key predictors

3. **Hyperparameter Optimization**
   - Expand search space: More hidden layer configurations, dropout rates
   - Implement RandomizedSearchCV for broader exploration
   - Use Bayesian optimization (e.g., Optuna) for efficient search
   - Increase max_iter to 300-500 with patience tuning

4. **Architecture Improvements**
   - Experiment with deeper networks (3-4 hidden layers)
   - Add dropout layers for regularization
   - Test batch normalization between layers
   - Implement learning rate scheduling

5. **Ensemble Methods**
   - Combine MLP with tree-based models (Random Forest, XGBoost)
   - Use stacking or voting classifiers
   - Implement model averaging across multiple seeds

### Code Quality

1. **Modularization**
   - Extract preprocessing steps into reusable functions
   - Create separate modules for data loading, preprocessing, modeling, and evaluation
   - Implement configuration files for hyperparameters

2. **Validation Strategy**
   - Implement nested cross-validation for unbiased performance estimates
   - Add time-series aware splitting if temporal patterns exist
   - Include calibration analysis for probability predictions

3. **Documentation**
   - Add docstrings to all functions
   - Include inline comments for complex operations
   - Document assumptions and design decisions

4. **Error Handling**
   - Add comprehensive try-except blocks
   - Validate data types and shapes throughout pipeline
   - Implement input validation for user-facing functions

### Experimental Rigor

1. **Statistical Testing**
   - Perform statistical significance tests between models
   - Report confidence intervals for metrics
   - Conduct sensitivity analysis on key hyperparameters

2. **Fairness Analysis**
   - Evaluate model performance across demographic subgroups
   - Check for bias in predictions by race, sex, and age
   - Implement fairness constraints if disparities are detected

3. **Interpretability**
   - Add SHAP values for local explanations
   - Implement partial dependence plots
   - Create decision boundary visualizations for key feature pairs

4. **Computational Efficiency**
   - Profile code to identify bottlenecks
   - Implement early stopping with validation monitoring
   - Consider using GPU acceleration for larger experiments

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
```

## Usage

1. Install required dependencies
2. Place data files in the working directory
3. Run all cells sequentially in the Jupyter notebook
4. Review visualizations and performance metrics
5. Check output CSV for final predictions

## Authors

Group 11


William Donnell-Lonon
Emma Ewing

## License

This project is part of an academic assignment for machine learning coursework.
