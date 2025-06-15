"""
Model Interpretation Module

This module provides tools for interpreting machine learning models using:
- SHAP (SHapley Additive exPlanations) values
- Feature importance analysis
- Partial dependence plots
- Individual prediction explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import BaseEstimator
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_feature_importance(model: BaseEstimator,
                             X: pd.DataFrame,
                             feature_names: List[str],
                             output_dir: str,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Analyze and visualize feature importance.

    Args:
        model: Trained model with feature_importances_ attribute
        X: Feature matrix
        feature_names: List of feature names
        output_dir: Directory to save plots
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importance scores
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(top_n),
                x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance.png')
    plt.close()

    return feature_importance

def compute_shap_values(model: BaseEstimator,
                       X: pd.DataFrame,
                       output_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Compute and visualize SHAP values.

    Args:
        model: Trained model
        X: Feature matrix
        output_dir: Directory to save plots

    Returns:
        Tuple of SHAP values and feature names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # If output is a list (for RandomForestClassifier), take first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Create SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary.png')
    plt.close()

    # Create SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path / 'shap_importance.png')
    plt.close()

    return shap_values, X.columns.tolist()

def analyze_feature_interactions(shap_values: np.ndarray,
                               X: pd.DataFrame,
                               output_dir: str,
                               top_n: int = 3) -> Dict[str, float]:
    """
    Analyze feature interactions using SHAP interaction values.

    Args:
        shap_values: Computed SHAP values
        X: Feature matrix
        output_dir: Directory to save plots
        top_n: Number of top interactions to analyze

    Returns:
        Dictionary of interaction strengths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    interactions = np.zeros((n_features, n_features))

    # Compute interaction strengths
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction = np.abs(shap_values[:, i] * shap_values[:, j]).mean()
            interactions[i, j] = interaction
            interactions[j, i] = interaction

    # Create interaction matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(interactions,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='viridis')
    plt.title('Feature Interaction Strengths')
    plt.tight_layout()
    plt.savefig(output_path / 'feature_interactions.png')
    plt.close()

    # Get top interactions
    interaction_pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction_pairs.append((
                feature_names[i],
                feature_names[j],
                interactions[i, j]
            ))

    top_interactions = sorted(interaction_pairs,
                            key=lambda x: x[2],
                            reverse=True)[:top_n]

    return {f"{pair[0]}_{pair[1]}": pair[2] for pair in top_interactions}

def explain_prediction(model: BaseEstimator,
                      X: pd.DataFrame,
                      instance_idx: int,
                      output_dir: str) -> Dict[str, float]:
    """
    Generate explanation for a single prediction.

    Args:
        model: Trained model
        X: Feature matrix
        instance_idx: Index of instance to explain
        output_dir: Directory to save plots

    Returns:
        Dictionary of feature contributions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get instance to explain
    instance = X.iloc[instance_idx:instance_idx+1]

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)

    # If output is a list (for RandomForestClassifier), take first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Create force plot
    plt.figure()
    force_plot = shap.force_plot(
        explainer.expected_value[0] if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
        shap_values[0],
        instance.iloc[0],
        show=False
    )
    shap.save_html(output_path / 'force_plot.html', force_plot)

    # Get feature contributions
    contributions = dict(zip(X.columns, shap_values[0]))
    
    # Sort contributions by absolute value
    contributions = dict(sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    ))

    return contributions

def main():
    """
    Main function to demonstrate model interpretation.
    """
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create output directory
    output_dir = 'results/model_interpretation'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Analyze feature importance
    importance_df = analyze_feature_importance(
        model, X_test, X.columns.tolist(), output_dir
    )
    logger.info("\nFeature Importance:")
    logger.info(importance_df.head())

    # Compute SHAP values
    shap_values, feature_names = compute_shap_values(model, X_test, output_dir)
    
    # Analyze feature interactions
    interactions = analyze_feature_interactions(shap_values, X_test, output_dir)
    logger.info("\nTop Feature Interactions:")
    logger.info(interactions)

    # Explain a single prediction
    explanation = explain_prediction(model, X_test, 0, output_dir)
    logger.info("\nPrediction Explanation:")
    logger.info(dict(list(explanation.items())[:5]))

if __name__ == "__main__":
    main() 