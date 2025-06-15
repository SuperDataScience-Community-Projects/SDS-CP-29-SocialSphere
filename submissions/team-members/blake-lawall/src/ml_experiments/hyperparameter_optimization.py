"""
Hyperparameter Optimization Module

This module implements hyperparameter optimization using Optuna for:
- Classification models (Conflict prediction)
- Regression models (Addiction score prediction)
- Clustering models (Student segmentation)

It includes:
- Optuna trials with cross-validation
- MLflow integration for tracking trials
- Visualization of optimization results
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_classifier(X: pd.DataFrame,
                      y: pd.Series,
                      n_trials: int = 100,
                      cv: int = 5) -> Dict:
    """
    Optimize hyperparameters for classification model.

    Args:
        X: Feature matrix
        y: Target variable
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds

    Returns:
        Dictionary with best parameters and scores
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features',
                                                    ['sqrt', 'log2', None])
        }

        clf = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('f1_mean', scores.mean())
            mlflow.log_metric('f1_std', scores.std())
        
        return scores.mean()

    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

def optimize_regressor(X: pd.DataFrame,
                      y: pd.Series,
                      n_trials: int = 100,
                      cv: int = 5) -> Dict:
    """
    Optimize hyperparameters for regression model.

    Args:
        X: Feature matrix
        y: Target variable
        n_trials: Number of optimization trials
        cv: Number of cross-validation folds

    Returns:
        Dictionary with best parameters and scores
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0)
        }

        reg = RandomForestRegressor(**params, random_state=42)
        scores = -cross_val_score(reg, X, y, cv=cv,
                                scoring='neg_mean_squared_error')
        rmse = np.sqrt(scores.mean())
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('rmse_std', scores.std())
        
        return rmse

    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

def optimize_clustering(X: pd.DataFrame,
                      n_clusters_range: Tuple[int, int] = (2, 10),
                      n_trials: int = 50) -> Dict:
    """
    Optimize hyperparameters for clustering model.

    Args:
        X: Feature matrix
        n_clusters_range: Range of number of clusters to try
        n_trials: Number of optimization trials

    Returns:
        Dictionary with best parameters and scores
    """
    def objective(trial):
        params = {
            'n_clusters': trial.suggest_int('n_clusters',
                                          n_clusters_range[0],
                                          n_clusters_range[1]),
            'init': trial.suggest_categorical('init',
                                            ['k-means++', 'random']),
            'n_init': trial.suggest_int('n_init', 5, 15)
        }

        kmeans = KMeans(**params, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric('silhouette_score', score)
        
        return score

    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }

def visualize_optimization_results(study: optuna.Study,
                                 output_dir: str):
    """
    Visualize optimization results.

    Args:
        study: Completed Optuna study
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(output_path / 'optimization_history.png')
    plt.close()

    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(output_path / 'parameter_importance.png')
    plt.close()

    # Plot parallel coordinate
    plt.figure(figsize=(12, 6))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(output_path / 'parallel_coordinate.png')
    plt.close()

def main():
    """
    Main function to demonstrate hyperparameter optimization.
    """
    from sklearn.datasets import make_classification, make_regression
    from sklearn.preprocessing import StandardScaler

    # Create output directory
    output_dir = 'results/hyperparameter_optimization'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Classification example
    logger.info("Running classification optimization...")
    X_clf, y_clf = make_classification(n_samples=1000,
                                     n_features=20,
                                     n_informative=15)
    X_clf = StandardScaler().fit_transform(X_clf)
    clf_results = optimize_classifier(X_clf, y_clf, n_trials=20)
    logger.info("Best classification parameters:")
    logger.info(clf_results['best_params'])
    logger.info(f"Best F1 score: {clf_results['best_score']:.3f}")

    # Regression example
    logger.info("\nRunning regression optimization...")
    X_reg, y_reg = make_regression(n_samples=1000,
                                  n_features=20,
                                  n_informative=15)
    X_reg = StandardScaler().fit_transform(X_reg)
    reg_results = optimize_regressor(X_reg, y_reg, n_trials=20)
    logger.info("Best regression parameters:")
    logger.info(reg_results['best_params'])
    logger.info(f"Best RMSE: {reg_results['best_score']:.3f}")

    # Clustering example
    logger.info("\nRunning clustering optimization...")
    clustering_results = optimize_clustering(X_clf, n_trials=10)
    logger.info("Best clustering parameters:")
    logger.info(clustering_results['best_params'])
    logger.info(f"Best silhouette score: {clustering_results['best_score']:.3f}")

    # Visualize results
    for name, results in [('classification', clf_results),
                         ('regression', reg_results),
                         ('clustering', clustering_results)]:
        output_subdir = output_dir / name
        visualize_optimization_results(results['study'], str(output_subdir))

if __name__ == "__main__":
    main() 