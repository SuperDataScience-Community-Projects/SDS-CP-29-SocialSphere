"""
Model Ensembling Module

This module implements various ensembling techniques:
- Voting/Averaging: Combine predictions from multiple models
- Stacking: Train a meta-model on base model predictions
- Bagging: Bootstrap aggregating with random subsets
- Weighted Ensembling: Combine models with learned weights
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import mlflow
from typing import List, Dict, Union, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Weighted ensemble of classifiers."""
    
    def __init__(self, models: List[BaseEstimator], weights: List[float] = None):
        """
        Initialize weighted ensemble.

        Args:
            models: List of classifier models
            weights: List of model weights (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit each base model."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted probability predictions."""
        probas = [model.predict_proba(X) for model in self.models]
        return np.average(probas, axis=0, weights=self.weights)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get class predictions."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Weighted ensemble of regressors."""
    
    def __init__(self, models: List[BaseEstimator], weights: List[float] = None):
        """
        Initialize weighted ensemble.

        Args:
            models: List of regressor models
            weights: List of model weights (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1/len(models)] * len(models)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit each base model."""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get weighted predictions."""
        predictions = [model.predict(X) for model in self.models]
        return np.average(predictions, axis=0, weights=self.weights)

class StackingEnsemble(BaseEstimator):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self,
                 base_models: List[BaseEstimator],
                 meta_model: BaseEstimator,
                 use_proba: bool = False,
                 cv: int = 5):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of base models
            meta_model: Meta-learner model
            use_proba: Whether to use probability predictions
            cv: Number of cross-validation folds
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
        self.cv = cv
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit stacking ensemble.
        
        1. Generate cross-validation predictions from base models
        2. Train meta-model on base model predictions
        3. Retrain base models on full dataset
        """
        # Generate meta-features
        meta_features = self._get_meta_features(X, y)
        
        # Fit meta model
        self.meta_model.fit(meta_features, y)
        
        # Fit base models on full dataset
        for model in self.base_models:
            model.fit(X, y)
            
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions."""
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] if self.use_proba else model.predict(X)
            for model in self.base_models
        ])
        return self.meta_model.predict(meta_features)
    
    def _get_meta_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Generate meta-features using cross-validation."""
        meta_features = np.column_stack([
            cross_val_predict(
                model, X, y,
                cv=self.cv,
                method='predict_proba' if self.use_proba else 'predict'
            )[:, 1] if self.use_proba else
            cross_val_predict(model, X, y, cv=self.cv)
            for model in self.base_models
        ])
        return meta_features

def optimize_ensemble_weights(models: List[BaseEstimator],
                           X: pd.DataFrame,
                           y: pd.Series,
                           is_classifier: bool = True) -> List[float]:
    """
    Optimize ensemble weights using validation performance.

    Args:
        models: List of fitted models
        X: Validation features
        y: Validation targets
        is_classifier: Whether models are classifiers

    Returns:
        List of optimized weights
    """
    if is_classifier:
        predictions = np.column_stack([
            model.predict_proba(X)[:, 1] for model in models
        ])
    else:
        predictions = np.column_stack([
            model.predict(X) for model in models
        ])
    
    # Use Ridge regression to find optimal weights
    if is_classifier:
        weight_model = Ridge(alpha=1.0, positive=True)
    else:
        weight_model = Ridge(alpha=1.0, positive=True)
    
    weight_model.fit(predictions, y)
    weights = weight_model.coef_
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    return weights.tolist()

def create_and_evaluate_ensembles(X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                is_classifier: bool = True) -> Dict:
    """
    Create and evaluate different ensemble methods.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        is_classifier: Whether to use classification models

    Returns:
        Dictionary of ensemble results
    """
    with mlflow.start_run(run_name="ensemble_evaluation"):
        # Create base models
        if is_classifier:
            base_models = [
                RandomForestClassifier(n_estimators=100, random_state=i)
                for i in range(3)
            ]
            meta_model = LogisticRegression()
            metric = accuracy_score
        else:
            base_models = [
                RandomForestRegressor(n_estimators=100, random_state=i)
                for i in range(3)
            ]
            meta_model = Ridge()
            metric = r2_score
        
        # Train base models
        for i, model in enumerate(base_models):
            model.fit(X_train, y_train)
            score = metric(y_test, model.predict(X_test))
            mlflow.log_metric(f"base_model_{i}_score", score)
        
        # Create and evaluate ensembles
        results = {}
        
        # 1. Simple averaging
        if is_classifier:
            ensemble = WeightedEnsembleClassifier(base_models)
        else:
            ensemble = WeightedEnsembleRegressor(base_models)
        
        ensemble.fit(X_train, y_train)
        score = metric(y_test, ensemble.predict(X_test))
        results['average'] = score
        mlflow.log_metric("average_ensemble_score", score)
        
        # 2. Weighted ensemble
        weights = optimize_ensemble_weights(
            base_models, X_train, y_train, is_classifier
        )
        if is_classifier:
            weighted_ensemble = WeightedEnsembleClassifier(base_models, weights)
        else:
            weighted_ensemble = WeightedEnsembleRegressor(base_models, weights)
        
        weighted_ensemble.fit(X_train, y_train)
        score = metric(y_test, weighted_ensemble.predict(X_test))
        results['weighted'] = score
        mlflow.log_metric("weighted_ensemble_score", score)
        
        # 3. Stacking
        stacking = StackingEnsemble(
            base_models,
            meta_model,
            use_proba=is_classifier
        )
        stacking.fit(X_train, y_train)
        score = metric(y_test, stacking.predict(X_test))
        results['stacking'] = score
        mlflow.log_metric("stacking_ensemble_score", score)
        
        return results

def main():
    """Main function to demonstrate ensemble methods."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    # Classification example
    logger.info("Running classification ensemble example...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf_results = create_and_evaluate_ensembles(
        X_train, y_train, X_test, y_test, is_classifier=True
    )
    logger.info("Classification Results:")
    for method, score in clf_results.items():
        logger.info(f"{method}: {score:.3f}")
    
    # Regression example
    logger.info("\nRunning regression ensemble example...")
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    reg_results = create_and_evaluate_ensembles(
        X_train, y_train, X_test, y_test, is_classifier=False
    )
    logger.info("Regression Results:")
    for method, score in reg_results.items():
        logger.info(f"{method}: {score:.3f}")

if __name__ == "__main__":
    main() 