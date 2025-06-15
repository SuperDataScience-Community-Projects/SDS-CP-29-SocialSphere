"""
MLflow Experiment Tracking Module for Social Media Addiction Analysis

This module implements experiment tracking using MLflow for various machine learning tasks:
1. Binary Classification (Conflict Prediction)
2. Regression (Addiction Score Prediction)
3. Clustering (Student Segmentation)

The module includes:
- Experiment tracking with MLflow
- Model training and evaluation
- Hyperparameter tuning
- Feature importance analysis
- Model persistence
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up MLflow experiment
EXPERIMENT_NAME = "social_media_addiction_analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for ML models.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        Tuple[pd.DataFrame, List[str]]: Processed dataframe and feature names
    """
    # Select relevant features
    feature_cols = [
        'Age', 'Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
        'Mental_Health_Score', 'Academic_Performance',
        'Preferred_Social_Media', 'Gender', 'Academic_Level'
    ]
    
    # Create copy of dataframe with selected features
    df_features = df[feature_cols].copy()
    
    # Encode categorical variables
    categorical_cols = ['Preferred_Social_Media', 'Gender', 'Academic_Level']
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols)
    
    return df_encoded, df_encoded.columns.tolist()

def train_conflict_classifier(X_train: pd.DataFrame, 
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            model_params: Dict) -> Dict[str, float]:
    """
    Train and evaluate conflict classifier.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features
        y_train, y_test (pd.Series): Training and test labels
        model_params (Dict): Model hyperparameters

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    with mlflow.start_run(run_name="conflict_classifier"):
        # Log parameters
        mlflow.log_params(model_params)
        
        # Train model
        clf = RandomForestClassifier(**model_params, random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(clf, "conflict_classifier")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_artifact(feature_importance.to_csv('feature_importance.csv'))
        
        return metrics

def train_addiction_regressor(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series,
                            model_params: Dict) -> Dict[str, float]:
    """
    Train and evaluate addiction score regressor.

    Args:
        X_train, X_test (pd.DataFrame): Training and test features
        y_train, y_test (pd.Series): Training and test labels
        model_params (Dict): Model hyperparameters

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    with mlflow.start_run(run_name="addiction_regressor"):
        # Log parameters
        mlflow.log_params(model_params)
        
        # Train model
        reg = RandomForestRegressor(**model_params, random_state=42)
        reg.fit(X_train, y_train)
        
        # Make predictions
        y_pred = reg.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(reg, "addiction_regressor")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_artifact(feature_importance.to_csv('feature_importance.csv'))
        
        return metrics

def perform_clustering(X: pd.DataFrame,
                      n_clusters: int = 4,
                      method: str = 'kmeans') -> Tuple[np.ndarray, float]:
    """
    Perform clustering analysis.

    Args:
        X (pd.DataFrame): Input features
        n_clusters (int): Number of clusters for K-Means
        method (str): Clustering method ('kmeans' or 'hdbscan')

    Returns:
        Tuple[np.ndarray, float]: Cluster labels and silhouette score
    """
    with mlflow.start_run(run_name=f"clustering_{method}"):
        if method == 'kmeans':
            # Log parameters
            mlflow.log_param("n_clusters", n_clusters)
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X, labels)
            
            # Log metrics
            mlflow.log_metric("silhouette_score", silhouette)
            
            # Log model
            mlflow.sklearn.log_model(kmeans, "kmeans_model")
            
        elif method == 'hdbscan':
            # Fit HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
            labels = clusterer.fit_predict(X)
            
            # Calculate silhouette score (excluding noise points)
            mask = labels != -1
            if mask.any():
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = 0.0
            
            # Log metrics
            mlflow.log_metric("silhouette_score", silhouette)
            
        return labels, silhouette

def tune_hyperparameters(X: pd.DataFrame,
                        y: pd.Series,
                        model_type: str,
                        param_grid: Dict,
                        cv: int = 5) -> Dict:
    """
    Perform hyperparameter tuning.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        model_type (str): Type of model ('classifier' or 'regressor')
        param_grid (Dict): Grid of parameters to search
        cv (int): Number of cross-validation folds

    Returns:
        Dict: Best parameters found
    """
    with mlflow.start_run(run_name=f"hyperparameter_tuning_{model_type}"):
        # Select model based on type
        if model_type == 'classifier':
            model = RandomForestClassifier(random_state=42)
            scoring = 'f1'
        else:
            model = RandomForestRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        
        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)
        
        return grid_search.best_params_

def main():
    """
    Main function to run ML experiments.
    """
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_students_data.csv')
    
    # Prepare features
    X, feature_names = prepare_features(df)
    
    # Prepare target variables
    y_conflict = (df['Conflicts_Over_Social_Media'] >= 3).astype(int)
    y_addiction = df['Addiction_Score']
    
    # Split data
    X_train, X_test, y_conflict_train, y_conflict_test = train_test_split(
        X, y_conflict, test_size=0.2, random_state=42
    )
    _, _, y_addiction_train, y_addiction_test = train_test_split(
        X, y_addiction, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define model parameters
    classifier_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
    
    regressor_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 5
    }
    
    # Train and evaluate classifier
    classifier_metrics = train_conflict_classifier(
        X_train_scaled, y_conflict_train,
        X_test_scaled, y_conflict_test,
        classifier_params
    )
    logger.info("Classifier Metrics:")
    logger.info(classifier_metrics)
    
    # Train and evaluate regressor
    regressor_metrics = train_addiction_regressor(
        X_train_scaled, y_addiction_train,
        X_test_scaled, y_addiction_test,
        regressor_params
    )
    logger.info("Regressor Metrics:")
    logger.info(regressor_metrics)
    
    # Perform clustering
    cluster_labels, silhouette = perform_clustering(X_train_scaled)
    logger.info(f"Clustering Silhouette Score: {silhouette:.3f}")
    
    # Define parameter grid for tuning
    classifier_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    
    # Tune classifier hyperparameters
    best_params = tune_hyperparameters(
        X_train_scaled, y_conflict_train,
        'classifier', classifier_param_grid
    )
    logger.info("Best Classifier Parameters:")
    logger.info(best_params)

if __name__ == "__main__":
    main() 