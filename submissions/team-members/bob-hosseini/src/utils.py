"""
Utility functions for social media conflict classification project.

This module contains reusable functions for data preprocessing, model training,
evaluation, and visualization used in the classification notebooks.
"""

# Standard library imports
import json

# Third-party imports
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.models.signature import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

# ================== Data preprocessing ==================
def create_binary_conflict(df, target_column='Conflicts', threshold=None, visualize=True):
    """
    Create binary conflict classification (High vs Low) from conflict scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str, default='Conflicts'
        Name of the conflict column
    threshold : int or float, optional
        Threshold for binary classification. If None, uses median.
    visualize : bool, default=True
        Whether to create visualizations
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added binary conflict column
    dict
        Dictionary with analysis results including threshold, counts, and imbalance ratio
    """
    # Make a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Determine threshold
    if threshold is None:
        threshold = df_copy[target_column].median()
    
    # Create binary target variable
    df_copy['Conflict_Binary'] = df_copy[target_column].apply(
        lambda x: 'High' if x > threshold else 'Low'
    )
    
    # Calculate statistics
    conflict_counts = df_copy['Conflict_Binary'].value_counts()
    imbalance_ratio = conflict_counts.min() / conflict_counts.max() * 100
    
    # Print analysis
    print(f"Binary Conflict Classification:")
    print(f"Threshold: {threshold}")
    print(f"Low Conflict (0-{threshold}): {conflict_counts.get('Low', 0)} samples")
    print(f"High Conflict ({threshold+1}-max): {conflict_counts.get('High', 0)} samples")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}%")
    print(f"Class proportions:")
    print(conflict_counts / len(df_copy))

        # convert to 0 and 1
    df_copy['Conflict_Binary'] = df_copy['Conflict_Binary'].map({'Low': 0, 'High': 1})
    # print the first 5 rows of the binary variable
    # print(df_copy['Conflict_Binary'].head())
    
    # Create visualizations if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original distribution with threshold line
        axes[0].hist(df_copy[target_column], bins=range(int(df_copy[target_column].min()), 
                                                       int(df_copy[target_column].max()) + 2), 
                    alpha=0.7, edgecolor='black')
        axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold}')
        axes[0].set_xlabel('Conflict Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Original Conflict Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Binary distribution
        sns.countplot(data=df_copy, x='Conflict_Binary', ax=axes[1])
        axes[1].set_title(f'Binary Conflict Distribution\n(Threshold: {threshold})\n(0: Low, 1: High)')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, bar in enumerate(axes[1].patches):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    # Prepare results dictionary
    results = {
        'threshold': threshold,
        'counts': conflict_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'proportions': (conflict_counts / len(df_copy)).to_dict()
    }
    
    return df_copy, results


# ================== Feature engineering ==================

def encode_onehot_with_reference(df, column_name, prefix=None):
    """One-hot encoding with smallest category as reference (dropped)"""
    counts = df[column_name].value_counts()
    smallest_category = counts.index[-1]  # Get smallest category
    
    print(f"\n Encoding {column_name} with prefix {prefix}")
    print(f"Reference category (dropped): {smallest_category}")
    
    # Create one-hot encoded columns
    dummies = pd.get_dummies(df[column_name], prefix=prefix or column_name, drop_first=False)
    # Drop smallest category as reference
    dummies = dummies.drop(f'{prefix or column_name}_{smallest_category}', axis=1)

    # print the first 5 rows of the encoded variable
    print(dummies.head())

   
    # Concatenate with original dataframe
    df = pd.concat([df, dummies], axis=1)

    # print samples of the other category
    print(f"\nSamples of the other category:")
    mask = df[column_name] == smallest_category
    print(pd.concat([df[mask][[column_name]], dummies[mask]], axis=1).head())

    return df, dummies



# Frequency-based encoding for Country and Platform
def encode_frequency(df, column_name):
    freq = df[column_name].value_counts().to_dict()
    df[f'{column_name}_freq_encoded'] = df[column_name].map(freq)

    # print the first 5 rows of the encoding mapping   
    print(f"\n{column_name} frequency encoding (top 5):")
    for country, freq in list(freq.items())[:5]:
        print(f"{country}: {freq}")

    # print the first 5 rows of the encoded variable
    print(f"\n{column_name} frequency encoded variable (top 5):")
    print(df[[f'{column_name}_freq_encoded']].head())

    # plot encoded variable distribution
    plt.figure(figsize=(8, 3))
    sns.countplot(data=df, x=f'{column_name}_freq_encoded')
    plt.title(f'Distribution of {column_name} Encoded Variable')
    plt.ylabel('Count')
    plt.show()

    return df

# 1. Dictionary mapping each country to its continent
country_to_continent = {
    # Africa
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ghana": "Africa",
    # Asia
    "Bangladesh": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Indonesia": "Asia", "Taiwan": "Asia", "Hong Kong": "Asia",
    "Singapore": "Asia", "UAE": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Qatar": "Asia", "Kuwait": "Asia",
    "Bahrain": "Asia", "Oman": "Asia", "Jordan": "Asia",
    "Lebanon": "Asia", "Iraq": "Asia", "Yemen": "Asia",
    "Syria": "Asia", "Afghanistan": "Asia", "Pakistan": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Maldives": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Armenia": "Asia",
    "Georgia": "Asia", "Azerbaijan": "Asia", "Cyprus": "Asia",
    # Europe
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Poland": "Europe",
    "Romania": "Europe", "Hungary": "Europe", "Czech Republic": "Europe",
    "Slovakia": "Europe", "Croatia": "Europe", "Serbia": "Europe",
    "Slovenia": "Europe", "Bulgaria": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Ukraine": "Europe",
    "Moldova": "Europe", "Belarus": "Europe", "Russia": "Europe",
    "Luxembourg": "Europe", "Monaco": "Europe", "Andorra": "Europe",
    "San Marino": "Europe", "Vatican City": "Europe",
    "Liechtenstein": "Europe", "Montenegro": "Europe", "Albania": "Europe",
    "North Macedonia": "Europe", "Kosovo": "Europe", "Bosnia": "Europe",
    # North America
    "USA": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Jamaica": "North America",
    "Trinidad": "North America", "Bahamas": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}

# 2. Function to map a country to its continent
def cont_map(country):
    """
    Return the continent for a given country.
    If the country is not in the mapping, returns 'Other'.
    """
    return country_to_continent.get(country, "Other")

def map_to_continent(df, visualize=False):
    df["Continent"] = df["Country"].apply(cont_map)
    if visualize:
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x='Continent')
        plt.title('Distribution of Continent Variable')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
    return df


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Transformer to group rare categories into "Other"
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups categories whose absolute count in the training data is below `min_count`
    into a single 'Other' category. Accepts X as a 1-D array/Series or 2-D array/DataFrame.
    """
    def __init__(self, min_count=30):
        self.min_count = min_count
        self.frequent_categories_ = set()

    def fit(self, X, y=None):
        # Flatten to 1-D
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        
        counts = series.value_counts()
        self.frequent_categories_ = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        arr = np.array(X)
        flat = arr.ravel()
        series = pd.Series(flat).fillna("Missing")
        out = series.where(series.isin(self.frequent_categories_), "Other")
        # Return as 2-D (n_samples, 1)
        return out.to_frame()
    

# Transformer to map country to continent
continent_dict = {
    # Africa
    "Egypt": "Africa", "Morocco": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Kenya": "Africa", "Ghana": "Africa",
    # Asia
    "Bangladesh": "Asia", "India": "Asia", "China": "Asia",
    "Japan": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Indonesia": "Asia", "Taiwan": "Asia", "Hong Kong": "Asia",
    "Singapore": "Asia", "UAE": "Asia", "Israel": "Asia",
    "Turkey": "Asia", "Qatar": "Asia", "Kuwait": "Asia",
    "Bahrain": "Asia", "Oman": "Asia", "Jordan": "Asia",
    "Lebanon": "Asia", "Iraq": "Asia", "Yemen": "Asia",
    "Syria": "Asia", "Afghanistan": "Asia", "Pakistan": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Maldives": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Armenia": "Asia",
    "Georgia": "Asia", "Azerbaijan": "Asia", "Cyprus": "Asia",
    # Europe
    "UK": "Europe", "Germany": "Europe", "France": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Poland": "Europe",
    "Romania": "Europe", "Hungary": "Europe", "Czech Republic": "Europe",
    "Slovakia": "Europe", "Croatia": "Europe", "Serbia": "Europe",
    "Slovenia": "Europe", "Bulgaria": "Europe", "Estonia": "Europe",
    "Latvia": "Europe", "Lithuania": "Europe", "Ukraine": "Europe",
    "Moldova": "Europe", "Belarus": "Europe", "Russia": "Europe",
    "Luxembourg": "Europe", "Monaco": "Europe", "Andorra": "Europe",
    "San Marino": "Europe", "Vatican City": "Europe",
    "Liechtenstein": "Europe", "Montenegro": "Europe", "Albania": "Europe",
    "North Macedonia": "Europe", "Kosovo": "Europe", "Bosnia": "Europe",
    # North America
    "USA": "North America", "Canada": "North America",
    "Mexico": "North America", "Costa Rica": "North America",
    "Panama": "North America", "Jamaica": "North America",
    "Trinidad": "North America", "Bahamas": "North America",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Ecuador": "South America", "Uruguay": "South America",
    "Paraguay": "South America", "Bolivia": "South America",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania"
}
# class to map country to continent
class CountryToContinentMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X may be a 2D numpy array or DataFrame with shape (n_samples, 1)
        # Flatten it to a 1D array/Series first:
        vals = X.values if hasattr(X, 'values') else X
        flat = vals.ravel()               # shape (n_samples,)
        mapped = pd.Series(flat).map(self.mapping)
        return mapped.fillna("Other").to_frame()



# Helper to extract feature names even when some transformers lack get_feature_names_out
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, cols in column_transformer.transformers_:
        # Skip dropped columns or remainder
        if transformer == 'drop' or name == 'remainder':
            continue

        # Normalize cols into a list
        input_cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]

        # If it's a pipeline, grab its last step
        tr = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer

        # Attempt to get feature names
        if hasattr(tr, 'get_feature_names_out'):
            try:
                # First try passing the original column names
                names = tr.get_feature_names_out(input_cols)
            except Exception:
                try:
                    # Fallback to no-arg version
                    names = tr.get_feature_names_out()
                except Exception:
                    # Final fallback: use the input column names
                    names = input_cols
        else:
            # Transformer has no naming method
            names = input_cols

        feature_names.extend(names)
    return feature_names


# ===============================
# Classification Pipelines
# ===============================

def mlflow_dataset(X_train_full, X_test):
    train_ds = mlflow.data.from_pandas(
        df=X_train_full,
        source="../data/data_cleaned.pickle",
        name="social_sphere_train_v1"
    )
    test_ds = mlflow.data.from_pandas(
        df=X_test,
        source="../data/data_cleaned.pickle",
        name="social_sphere_test_v1"
    )
    return {"train_ds": train_ds, "test_ds": test_ds}


def run_and_register_dummy_baseline(
    X_train_full,
    y_train_full,
    preprocessor,
    strategy: str,
    cv,
    scoring,
    dataset,
    registered_model_name: str = "conflict_baseline_dummy"
):
    """
    Runs cross‐validation for a single DummyClassifier strategy, logs metrics to MLflow,
    refits on the full training set, and registers the model.

    Parameters:
    - X_train_full, y_train_full: training data
    - preprocessor: sklearn Pipeline or ColumnTransformer for feature prep
    - strategy: one of DummyClassifier strategies ("most_frequent", "stratified", "uniform", "prior")
    - cv: cross‐validation splitter (e.g., StratifiedKFold instance)
    - experiment_name: MLflow experiment under which runs are logged
    - registered_model_name: name to register the final model in MLflow
    """
    
    run_name = f"baseline_dummy_{strategy}_cv"
    with mlflow.start_run(run_name=run_name):
        # Log inputs
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")

        # build pipeline
        baseline_pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", DummyClassifier(strategy=strategy))
        ])

        # log strategy
        mlflow.log_param("strategy", strategy)

        # cross‐validate
        cv_results = cross_validate(
            estimator=baseline_pipeline,
            X=X_train_full,
            y=y_train_full,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # log mean and std for each metric
        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            mlflow.log_metric(metric,      scores.mean())
            mlflow.log_metric(f"{metric}_std", scores.std())

        # raw fold scores as artifact
        with open("cv_fold_scores.json", "w") as fp:
            json.dump({k: v.tolist() for k, v in cv_results.items()}, fp)
        mlflow.log_artifact("cv_fold_scores.json")

        # refit on full training data
        baseline_pipeline.fit(X_train_full, y_train_full)

        # infer signature and example
        example_input = X_train_full.iloc[:5]
        example_preds = baseline_pipeline.predict(example_input)
        signature = infer_signature(example_input, example_preds)

        # log & register
        mlflow.sklearn.log_model(
            sk_model=baseline_pipeline,
            name=f"baseline_model_cv_{strategy}",
            registered_model_name=f"{registered_model_name}_{strategy}",
            signature=signature,
            input_example=example_input
        )


def run_classification_experiment(
    name: str,
    estimator,                # e.g. Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(...))])
    X_train, y_train,
    cv,
    scoring,            # dict of scoring metrics
    dataset,
    hparams,
    registered_model_name: str = "conflict_baseline_dummy"
):
    with mlflow.start_run(run_name=name):
        # Log inputs
        mlflow.log_input(dataset["train_ds"], context="training")    
        mlflow.log_input(dataset["test_ds"], context="test")

        # log strategy
        mlflow.log_param("hyperparameters", hparams)

        # 2) Cross-validate & log CV metrics
        cv_results = cross_validate(
            estimator=estimator,
            X=X_train, y=y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                mlflow.log_metric(metric.replace("test_", ""), scores.mean().round(2))

        # 3) Re-fit on full train and register model
        estimator.fit(X_train, y_train)

        # 4) Infer signature & input example for better model packaging
        example_input = X_train.iloc[:5]
        preds = estimator.predict(example_input)
        signature = infer_signature(example_input, preds)

        mlflow.sklearn.log_model(
            sk_model=estimator,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )
