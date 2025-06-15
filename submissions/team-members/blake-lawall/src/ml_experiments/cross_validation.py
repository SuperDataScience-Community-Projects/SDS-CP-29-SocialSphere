"""
Cross-Validation and Statistical Testing Module

This module implements:
- Stratified k-fold cross-validation
- Time series cross-validation
- Statistical significance tests
- Performance comparison between models
- Confidence intervals for metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.base import BaseEstimator, clone
from scipy import stats
import mlflow
from typing import List, Dict, Union, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_cross_validation(model: BaseEstimator,
                           X: pd.DataFrame,
                           y: pd.Series,
                           cv_type: str = 'stratified',
                           n_splits: int = 5,
                           scoring: Union[str, List[str]] = None) -> Dict:
    """
    Perform cross-validation with specified method.

    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target variable
        cv_type: Type of cross-validation ('stratified', 'kfold', 'timeseries')
        n_splits: Number of CV splits
        scoring: Scoring metric(s)

    Returns:
        Dictionary of CV results
    """
    # Set up cross-validation splitter
    if cv_type == 'stratified':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_type == 'timeseries':
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Log results to MLflow
    with mlflow.start_run(run_name=f"cv_{cv_type}"):
        for metric, values in cv_results.items():
            if metric.startswith(('test_', 'train_')):
                mlflow.log_metric(f"{metric}_mean", np.mean(values))
                mlflow.log_metric(f"{metric}_std", np.std(values))
    
    return cv_results

def compare_models(models: List[BaseEstimator],
                  X: pd.DataFrame,
                  y: pd.Series,
                  scoring: str,
                  cv: int = 5) -> Tuple[Dict, float]:
    """
    Compare multiple models using statistical tests.

    Args:
        models: List of models to compare
        X: Feature matrix
        y: Target variable
        scoring: Scoring metric
        cv: Number of CV folds

    Returns:
        Tuple of results dictionary and p-value
    """
    # Get cross-validation scores for each model
    cv_scores = []
    model_names = []
    
    for model in models:
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
        cv_scores.append(scores)
        model_names.append(model.__class__.__name__)
    
    # Perform Friedman test
    friedman_statistic, p_value = stats.friedmanchisquare(*cv_scores)
    
    # Create results dictionary
    results = {
        'model_names': model_names,
        'cv_scores': cv_scores,
        'mean_scores': [scores.mean() for scores in cv_scores],
        'std_scores': [scores.std() for scores in cv_scores],
        'friedman_statistic': friedman_statistic,
        'p_value': p_value
    }
    
    # Log results to MLflow
    with mlflow.start_run(run_name="model_comparison"):
        for name, mean_score, std_score in zip(
            model_names,
            results['mean_scores'],
            results['std_scores']
        ):
            mlflow.log_metric(f"{name}_mean", mean_score)
            mlflow.log_metric(f"{name}_std", std_score)
        mlflow.log_metric("friedman_p_value", p_value)
    
    return results, p_value

def calculate_confidence_intervals(scores: np.ndarray,
                                confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for CV scores.

    Args:
        scores: Array of CV scores
        confidence: Confidence level

    Returns:
        Tuple of lower and upper bounds
    """
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    
    # Calculate confidence interval
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * (std / np.sqrt(n))
    
    return mean - margin, mean + margin

def visualize_cv_results(results: Dict,
                        output_dir: str):
    """
    Visualize cross-validation results.

    Args:
        results: Dictionary of CV results
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot score distributions
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    
    for name, scores in zip(results['model_names'], results['cv_scores']):
        data.extend(scores)
        labels.extend([name] * len(scores))
    
    sns.boxplot(x=labels, y=data)
    plt.title('Cross-Validation Score Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'cv_distributions.png')
    plt.close()
    
    # Plot mean scores with confidence intervals
    plt.figure(figsize=(12, 6))
    means = results['mean_scores']
    stds = results['std_scores']
    
    plt.errorbar(
        range(len(means)),
        means,
        yerr=stds,
        fmt='o',
        capsize=5
    )
    plt.xticks(range(len(means)), results['model_names'], rotation=45)
    plt.title('Mean Scores with Standard Deviation')
    plt.tight_layout()
    plt.savefig(output_path / 'mean_scores.png')
    plt.close()

def main():
    """Main function to demonstrate CV and statistical testing."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
    
    # Create models
    models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        LogisticRegression(random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    # Create output directory
    output_dir = 'results/cross_validation'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Perform cross-validation for each model
    logger.info("Performing cross-validation...")
    cv_results = {}
    for model in models:
        name = model.__class__.__name__
        results = perform_cross_validation(
            model, X, y,
            scoring=['accuracy', 'f1', 'roc_auc']
        )
        cv_results[name] = results
        logger.info(f"\n{name} Results:")
        for metric, values in results.items():
            if metric.startswith('test_'):
                logger.info(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    # Compare models
    logger.info("\nComparing models...")
    comparison_results, p_value = compare_models(
        models, X, y, scoring='accuracy'
    )
    logger.info(f"Friedman test p-value: {p_value:.4f}")
    
    # Visualize results
    visualize_cv_results(comparison_results, output_dir)
    
    # Calculate confidence intervals
    logger.info("\nConfidence Intervals (95%):")
    for name, scores in zip(
        comparison_results['model_names'],
        comparison_results['cv_scores']
    ):
        lower, upper = calculate_confidence_intervals(scores)
        logger.info(f"{name}: ({lower:.3f}, {upper:.3f})")

if __name__ == "__main__":
    main() 