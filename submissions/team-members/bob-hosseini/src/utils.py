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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression
from scipy import sparse
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ======== Warnings ========
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)
warnings.filterwarnings(
    "ignore",
    message="Hint: Inferred schema contains integer column"
)



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

# The function to run a classification experiment
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




# ===============================
# SHAP
# ===============================
def run_shap_experiment(
    best_model,
    X_train_full,
    feature_perturbation="interventional",
    plot_type="bar",
    shap_type="linear"
):
    # 1. Extract best model and its preprocessing step
    # best_model = grid_search_result.best_estimator_
    preprocessor = best_model.named_steps['preprocessing']
    classifier   = best_model.named_steps['classifier']

    # 2. Prepare data for SHAP
    #    Use a subset of training data (or validation) for faster computation
    X_shap_raw = X_train_full.sample(n=200, random_state=42)
    # Transform to model inputs
    X_shap_proc = preprocessor.transform(X_shap_raw)
    if sparse.issparse(X_shap_proc):
        X_shap_proc = X_shap_proc.toarray()

    # Recover feature names
    feature_names = get_feature_names(preprocessor)
    X_shap_df = pd.DataFrame(X_shap_proc, columns=feature_names)

    if shap_type == "linear":    
        # Initialize a SHAP explainer
        explainer = shap.LinearExplainer(
            model=classifier,
            masker=X_shap_df,
            feature_perturbation=feature_perturbation   # supported: "interventional" or "correlation_dependent"
        )
    else:
        explainer = shap.TreeExplainer(
            model=classifier,              # the XGBClassifier instance
            data=X_shap_df,                # background data for expected values
            feature_perturbation=feature_perturbation  # optional
        )

    # Compute SHAP values
    shap_values = explainer.shap_values(X_shap_df)

    # Summary plot for the positive class (if binary)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_shap_df, plot_type=plot_type, show=False
    )
    plt.title("SHAP Summary Plot")
    # plt.show()
    fig_sum = plt.gcf()
    plt.close(fig_sum)
    return fig_sum

# ===============================
# log_test_set_performance
# ===============================

def log_test_set_performance(
    model,
    X_test,
    y_test,
    prefix: str = "test"
):
    """
    Evaluate `model` on (X_test, y_test), compute common metrics,
    a confusion matrix plot, and a text classification report,
    then log everything into the active MLflow run.
    """
    # 1) Predictions & probabilities
    y_pred  = model.predict(X_test)
    proba    = model.predict_proba(X_test)[:, 1]

    # 2) Compute metrics
    metrics = {
        f"{prefix}_accuracy":   accuracy_score(y_test, y_pred),
        f"{prefix}_precision":  precision_score(y_test, y_pred),
        f"{prefix}_recall":     recall_score(y_test, y_pred),
        f"{prefix}_f1_score":   f1_score(y_test, y_pred),
        f"{prefix}_roc_auc":    roc_auc_score(y_test, proba),
    }
    # 3) Log metrics
    for name, val in metrics.items():
        mlflow.log_metric(name, round(val, 2))

    # 4) Classification report
    report = classification_report(y_test, y_pred)
    report_path = f"{prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # 5) Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    cm_path = f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path)




# ===============================
# Grid Search
# ===============================

def run_classification_gridsearch_experiment(
    name: str,
    pipeline,               # e.g. Pipeline([... , ('classifier', LogisticRegression())])
    param_grid: dict,
    X_train, y_train, X_test, y_test,
    cv,                     # e.g. StratifiedKFold(...)
    scoring,                # e.g. {"accuracy":"accuracy", "f1_score":"f1", ...}
    dataset: dict,          # {"train_ds": X_train_full, "test_ds": X_test}
    registered_model_name: str,
    verbose: bool = False,
    feature_set: str = "all"
):
    grid_search = GridSearchCV(
        estimator=pipeline,        # your Pipeline([... ('classifier', LogisticRegression()) ])
        param_grid=param_grid,
        cv=cv,                         # 5-fold stratified CV
        scoring=scoring,            # primary metric
        n_jobs=-1,                    # parallelize across all cores
        return_train_score=True,
        refit='f1_score',            # ← what to optimize/return as best_estimator_
        error_score=np.nan,  # treat fold‐errors as NaN rather than crashing
        verbose=1
    )

    with mlflow.start_run(run_name=name):
        # 1) Log dataset inputs
        mlflow.log_input(dataset["train_ds"], context="training")
        mlflow.log_input(dataset["test_ds"],  context="test")

        # 2) Log the grid search hyperparameter space
        mlflow.log_param("grid_search_params", param_grid)
        
        # Log the feature set
        mlflow.log_param("feature_set", feature_set)

        grid_search.fit(X_train, y_train)

        # 4) Log the best hyperparameters
        best_params = grid_search.best_params_
        mlflow.log_param("hyperparameters", best_params)

        # 5) Log CV metrics (mean ± std) for each scoring key
        results = grid_search.cv_results_
        # if scoring is a dict, iterate its keys; else assume single metric
        for metric in scoring:
            mean_score = results[f"mean_test_{metric}"][grid_search.best_index_]
            std_score  = results[f"std_test_{metric}"][grid_search.best_index_]
            mlflow.log_metric(metric, mean_score.round(2))
            mlflow.log_metric(f"{metric}_std", std_score)
            if verbose:
                print(f"{metric}: {mean_score.round(2)} ± {std_score}")

        # 6) Log and register the best estimator
        best_estimator = grid_search.best_estimator_
        example_input = X_train.iloc[:5]
        example_preds = best_estimator.predict(example_input)
        signature = infer_signature(example_input, example_preds)
        
        # Adding shap to the model
        if 'baseline' not in name:
            if 'logreg' in name:
                shap_type = "linear"
            else:
                shap_type = "tree"
            fig_sum = run_shap_experiment(
                best_model=best_estimator,
                X_train_full=X_train,
                feature_perturbation="interventional",
                plot_type="violin", # other options: "bar", "dot", "violin"
                shap_type=shap_type
            )
            mlflow.log_figure(fig_sum, "shap_summary.png")

        # Log test set performance
        log_test_set_performance(
            model=best_estimator,
            X_test=X_test,
            y_test=y_test,
            prefix="test"
        )

        # 7) Log the best estimator
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name=name,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=example_input
        )
    if verbose:
        # 3. Print summary in the output cell
        print("Best parameters:", grid_search.best_params_)
        print("Best F1 score (CV):", grid_search.best_score_)

        # 4. (Optional) inspect all CV results
        cv_df = pd.DataFrame(grid_search.cv_results_)
        display(cv_df.sort_values("mean_test_f1_score", ascending=False).head(3))
    return best_estimator, grid_search