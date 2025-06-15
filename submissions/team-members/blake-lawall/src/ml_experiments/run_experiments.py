"""
Run ML Experiments Script

This script orchestrates the execution of all machine learning experiments,
including classification, regression, and clustering tasks.
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    CLASSIFIER_PARAMS,
    REGRESSOR_PARAMS,
    CLASSIFIER_PARAM_GRID,
    REGRESSOR_PARAM_GRID,
    CLUSTERING_PARAMS,
    DATA_DIR,
    RESULTS_DIR
)
from experiment_tracking import (
    prepare_features,
    train_conflict_classifier,
    train_addiction_regressor,
    perform_clustering,
    tune_hyperparameters
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")

def load_and_prepare_data():
    """Load and prepare data for experiments."""
    logger.info("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(DATA_DIR / "preprocessed_students_data.csv")
    
    # Prepare features and targets
    X, feature_names = prepare_features(df)
    y_conflict = (df['Conflicts_Over_Social_Media'] >= 3).astype(int)
    y_addiction = df['Addiction_Score']
    
    logger.info(f"Data loaded: {len(df)} samples, {len(feature_names)} features")
    return X, y_conflict, y_addiction, feature_names

def run_classification_experiments(X, y):
    """Run classification experiments."""
    logger.info("\nRunning classification experiments...")
    
    # First, tune hyperparameters
    logger.info("Tuning classifier hyperparameters...")
    best_params = tune_hyperparameters(
        X, y, 'classifier', CLASSIFIER_PARAM_GRID
    )
    logger.info(f"Best parameters: {best_params}")
    
    # Train with best parameters
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    metrics = train_conflict_classifier(
        X_train, y_train,
        X_test, y_test,
        best_params
    )
    
    logger.info("Classification metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3f}")
    
    return metrics, best_params

def run_regression_experiments(X, y):
    """Run regression experiments."""
    logger.info("\nRunning regression experiments...")
    
    # First, tune hyperparameters
    logger.info("Tuning regressor hyperparameters...")
    best_params = tune_hyperparameters(
        X, y, 'regressor', REGRESSOR_PARAM_GRID
    )
    logger.info(f"Best parameters: {best_params}")
    
    # Train with best parameters
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    metrics = train_addiction_regressor(
        X_train, y_train,
        X_test, y_test,
        best_params
    )
    
    logger.info("Regression metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.3f}")
    
    return metrics, best_params

def run_clustering_experiments(X):
    """Run clustering experiments."""
    logger.info("\nRunning clustering experiments...")
    
    results = {}
    for method in ['kmeans', 'hdbscan']:
        logger.info(f"Running {method} clustering...")
        labels, silhouette = perform_clustering(
            X,
            n_clusters=CLUSTERING_PARAMS['kmeans']['n_clusters'],
            method=method
        )
        
        results[method] = {
            'labels': labels,
            'silhouette': silhouette
        }
        
        logger.info(f"{method.upper()} Silhouette Score: {silhouette:.3f}")
        if method == 'kmeans':
            unique_labels = np.unique(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                logger.info(f"Cluster {label}: {count} samples")
    
    return results

def save_results(classification_results, regression_results, clustering_results):
    """Save experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save classification results
    with open(results_dir / "classification_results.txt", "w") as f:
        f.write("Classification Results\n")
        f.write("=====================\n\n")
        f.write("Metrics:\n")
        for metric, value in classification_results[0].items():
            f.write(f"{metric}: {value:.3f}\n")
        f.write("\nBest Parameters:\n")
        for param, value in classification_results[1].items():
            f.write(f"{param}: {value}\n")
    
    # Save regression results
    with open(results_dir / "regression_results.txt", "w") as f:
        f.write("Regression Results\n")
        f.write("=================\n\n")
        f.write("Metrics:\n")
        for metric, value in regression_results[0].items():
            f.write(f"{metric}: {value:.3f}\n")
        f.write("\nBest Parameters:\n")
        for param, value in regression_results[1].items():
            f.write(f"{param}: {value}\n")
    
    # Save clustering results
    with open(results_dir / "clustering_results.txt", "w") as f:
        f.write("Clustering Results\n")
        f.write("=================\n\n")
        for method, results in clustering_results.items():
            f.write(f"{method.upper()} Results:\n")
            f.write(f"Silhouette Score: {results['silhouette']:.3f}\n")
            if method == 'kmeans':
                unique_labels = np.unique(results['labels'])
                f.write("Cluster Sizes:\n")
                for label in unique_labels:
                    count = np.sum(results['labels'] == label)
                    f.write(f"Cluster {label}: {count} samples\n")
            f.write("\n")
    
    logger.info(f"Results saved to {results_dir}")

def main():
    """Main function to run all experiments."""
    # Setup MLflow
    setup_mlflow()
    
    # Load and prepare data
    X, y_conflict, y_addiction, feature_names = load_and_prepare_data()
    
    # Run experiments
    classification_results = run_classification_experiments(X, y_conflict)
    regression_results = run_regression_experiments(X, y_addiction)
    clustering_results = run_clustering_experiments(X)
    
    # Save results
    save_results(classification_results, regression_results, clustering_results)
    
    logger.info("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main() 