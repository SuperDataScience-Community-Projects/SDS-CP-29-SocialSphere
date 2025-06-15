"""
Configuration Module for ML Experiments

This module contains configuration settings for ML experiments,
including MLflow settings and model parameters.
"""

import os
from pathlib import Path

# MLflow settings
MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "social_media_addiction_analysis"

# Model parameters
CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

REGRESSOR_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'random_state': 42
}

# Hyperparameter tuning settings
CLASSIFIER_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

REGRESSOR_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

# Clustering parameters
CLUSTERING_PARAMS = {
    'kmeans': {
        'n_clusters': 4,
        'random_state': 42
    },
    'hdbscan': {
        'min_cluster_size': 10,
        'min_samples': 5
    }
}

# Data settings
FEATURE_COLUMNS = [
    'Age', 'Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Academic_Performance',
    'Preferred_Social_Media', 'Gender', 'Academic_Level'
]

CATEGORICAL_COLUMNS = [
    'Preferred_Social_Media', 'Gender', 'Academic_Level'
]

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 