# Comparative Analysis of Ridge Regression vs. Random Forest on Synthetic Datasets "Ensenada Baja California City Within Macro and Micro Economic Rules"

## 1. Research Overview & Objective
This project establishes a rigorous benchmarking framework to analyze the algorithmic efficiency, predictive power, and bias-variance trade-off between parametric regularization (**Ridge Regression**) and non-parametric ensemble architectures (**Random Forest**). 

Applied to high-variance synthetic datasets simulating MSME (*MiPyMEs*) financial performance and profitability constraints, this study evaluates model behavior under controlled multi-collinearity, feature sparsity, and non-linear economic perturbations.

## 2. Mathematical Foundation & Methodology

### Ridge Regression (L2 Regularization)
To mitigate multi-collinearity in dense economic indicators, the Ridge model introduces an $L_2$ penalty to the ordinary least squares (OLS) loss function:

$$\min_{\beta} \left| y - X\beta \right|^2_2 + \lambda \left| \beta \right|^2_2$$

Where $\lambda \ge 0$ controls the shrinkage intensity of the parameter vector $\beta$, forcing stable coefficient trajectories under high feature correlation.

### Random Forest (Ensemble Bagging)
The non-parametric architecture leverages an ensemble of $B$ uncorrelated decision trees grown on bootstrapped training samples. The final predictive function aggregates individual tree estimators to systematically reduce variance:

$$\hat{f}_{rf}^{B}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

## 3. Experimental Architecture & Technical Stack
The pipeline is engineered modularly in Python 3.10:
* **`src/simulation_engine.py`:** Executes Monte Carlo simulations to generate synthetic economic indicators with controlled covariance structures.
* **`src/model_pipeline.py`:** Handles automated hyperparameter tuning via cross-validation ($k$-fold), pipeline standardization, and multi-metric evaluation (MSE, MAE, $R^2$).

**Tech Stack:** Scikit-Learn, NumPy, Pandas, SciPy, Matplotlib, Seaborn.

## 4. Analytical Insights & Core Trade-offs
* **Parametric Constraints:** Ridge Regression demonstrated superior stability in strictly linear regimes with low $\lambda$ tuning, but suffered performance decay when handling non-linear interactions without explicit polynomial feature mappings.
* **Non-Linear Dominance:** Random Forest successfully captured high-order interactions and operational boundaries, showing high robustness against outliers at the cost of increased computational complexity and subtle overfitting risks in sparse feature regions.

## 5. Visualizations & Diagnostic Analytics

### Performance Benchmarking (Bias-Variance Trade-off)
The framework evaluates structural error dynamics across models. The comparison profiles how the parametric constraints of Ridge Regularization handle structural noise versus the high-capacity adaptability of Random Forest.

<p align="center">
  <img src="visualizations/bias_variance_tradeoff.png" width="600" alt="Bias Variance Trade Off">
</p>

### Feature Importance Profile
By extracting the architectural weights from the ensemble model, the pipeline identifies the structural drivers of enterprise scaling. The empirical results isolate the Digitalization Index as a primary predictor of long-term profitability, outperforming traditional metrics like enterprise seniority.

<p align="center">
  <img src="visualizations/feature_importance.png" width="600" alt="Feature Importance">
</p>

