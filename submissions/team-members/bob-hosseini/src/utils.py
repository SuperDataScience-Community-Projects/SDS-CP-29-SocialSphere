"""
Utility functions for social media conflict classification project.

This module contains reusable functions for data preprocessing, model training,
evaluation, and visualization used in the classification notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow
import mlflow.sklearn


def load_cleaned_data(file_path):
    """
    Load cleaned data from pickle file.
    
    Parameters:
    -----------
    file_path : str
        Path to the pickle file containing cleaned data
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    return df


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


def visualize_target_distribution(df, original_column, binary_column, threshold):
    """
    Visualize original and binary target distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    original_column : str
        Name of original target column
    binary_column : str
        Name of binary target column
    threshold : float
        Threshold value used for binary conversion
    """
    plt.figure(figsize=(10, 4))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    df[original_column].hist(bins=20, alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.xlabel(original_column.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.title('Original Target Distribution')
    plt.legend()
    
    # Binary distribution
    plt.subplot(1, 2, 2)
    df[binary_column].value_counts().plot(kind='bar')
    plt.xlabel('Conflict Level')
    plt.ylabel('Count')
    plt.title('Binary Target Distribution')
    plt.xticks([0, 1], ['Low', 'High'], rotation=0)
    
    plt.tight_layout()
    plt.show()

def downsample_majority_class(df, target_column):
    """
    Downsample the majority class to balance the dataset
    
    Parameters:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column (e.g., 'Conflict_Binary')
    
    Returns:
    pd.DataFrame: Balanced dataframe with downsampled majority class
    dict: Information about the downsampling process
    """
    from sklearn.utils import resample
    
    # Get class counts
    class_counts = df[target_column].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    
    # Calculate target size for majority class
    target_majority_size = int(minority_count)
    print(f"Target majority size: {target_majority_size}")
    
    # If target size is greater than current majority size, no downsampling needed
    if target_majority_size >= majority_count:
        print(f"No downsampling needed. Current ratio: {minority_count/majority_count:.2f}")
        return df, {
            'original_majority_count': majority_count,
            'original_minority_count': minority_count,
            'final_majority_count': majority_count,
            'final_minority_count': minority_count,
            'downsampled': False
        }
    
    # Separate majority and minority classes
    df_majority = df[df[target_column] == majority_class]
    df_minority = df[df[target_column] == minority_class]
    
    # Downsample majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,  # sample without replacement
        n_samples=target_majority_size,
        random_state=42
    )
    
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_minority, df_majority_downsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print results
    print(f"Original dataset:")
    print(f"  {majority_class}: {majority_count}")
    print(f"  {minority_class}: {minority_count}")
    print(f"  Ratio (minority/majority): {minority_count/majority_count:.2f}")
    
    print(f"\nBalanced dataset:")
    print(f"  {majority_class}: {target_majority_size}")
    print(f"  {minority_class}: {minority_count}")
    print(f"  Ratio (minority/majority): {minority_count/target_majority_size:.2f}")
    
    return df_balanced, {
        'original_majority_count': majority_count,
        'original_minority_count': minority_count,
        'final_majority_count': target_majority_size,
        'final_minority_count': minority_count,
        'downsampled': True,
        'samples_removed': majority_count - target_majority_size
    }

def identify_feature_types(X):
    """
    Identify numeric and categorical features in dataset.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
        
    Returns:
    --------
    tuple
        (numeric_features, categorical_features) as lists
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    return numeric_features, categorical_features


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


# Grouping Platforms with category size less than 10% of the total
def group_and_encode_features(df_data, target_column, threshold=30, visualize=True):
    """
    Group categorical features with low frequency into 'Other' category and apply one-hot encoding
    
    Parameters:
    df_data (pd.DataFrame): The input dataframe
    target_column (str): The name of the categorical column to group and encode
    threshold (int): Minimum count threshold for keeping categories separate (default: 30)
    visualize (bool): Whether to show visualizations (default: True)
    
    Returns:
    pd.DataFrame: DataFrame with grouped and one-hot encoded features
    dict: Summary information about the grouping process
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Make a copy to avoid modifying original data
    df_processed = df_data.copy()
    
    # Get original value counts
    original_counts = df_processed[target_column].value_counts()
    
    # Create grouped column name
    grouped_column = f"{target_column}_group"
    
    # Group categories with count less than threshold into 'Other'
    df_processed[grouped_column] = df_processed[target_column].apply(
        lambda x: 'Other' if original_counts[x] < threshold else x
    )
    
    # Get grouped value counts
    grouped_counts = df_processed[grouped_column].value_counts()
    
    # Print summary
    print(f"=== Feature Grouping Summary for '{target_column}' ===")
    print(f"Threshold: {threshold}")
    print(f"Original categories: {len(original_counts)}")
    print(f"Grouped categories: {len(grouped_counts)}")
    print(f"Categories moved to 'Other': {len(original_counts) - len(grouped_counts) + (1 if 'Other' in grouped_counts else 0)}")
    
    print(f"\nOriginal distribution:")
    for category, count in original_counts.head(10).items():
        status = "→ Other" if count < threshold else "→ Kept"
        print(f"  {category}: {count} {status}")
    if len(original_counts) > 10:
        print(f"  ... and {len(original_counts) - 10} more categories")
    
    print(f"\nGrouped distribution:")
    for category, count in grouped_counts.items():
        print(f"  {category}: {count}")
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution (top 10 categories)
        top_original = original_counts.head(10)
        axes[0].bar(range(len(top_original)), top_original.values)
        axes[0].set_title(f'Original Distribution of {target_column}\n(Top 10 categories)')
        axes[0].set_xlabel('Categories')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(top_original)))
        axes[0].set_xticklabels(top_original.index, rotation=45, ha='right')
        
        # Add threshold line
        axes[0].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
        axes[0].legend()
        
        # Grouped distribution
        axes[1].bar(range(len(grouped_counts)), grouped_counts.values)
        axes[1].set_title(f'Grouped Distribution of {grouped_column}')
        axes[1].set_xlabel('Categories')
        axes[1].set_ylabel('Count')
        axes[1].set_xticks(range(len(grouped_counts)))
        axes[1].set_xticklabels(grouped_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    # One-hot encode the grouped column
    df_encoded = pd.get_dummies(df_processed, columns=[grouped_column], prefix=target_column)
    
    # Get the new one-hot encoded columns
    encoded_columns = [col for col in df_encoded.columns if col.startswith(f'{target_column}_')]
    
    print(f"\nOne-hot encoded columns created:")
    for col in encoded_columns:
        print(f"  {col}")
    
    # Show sample of encoded features
    print(f"\nSample of encoded features:")
    print(df_encoded[encoded_columns].head())
    
    # Create summary dictionary
    summary = {
        'original_categories': len(original_counts),
        'grouped_categories': len(grouped_counts),
        'threshold_used': threshold,
        'categories_moved_to_other': len(original_counts) - len(grouped_counts) + (1 if 'Other' in grouped_counts else 0),
        'original_distribution': original_counts.to_dict(),
        'grouped_distribution': grouped_counts.to_dict(),
        'encoded_columns': encoded_columns,
        'grouped_column_name': grouped_column
    }
    
    return df_encoded, summary

# ================== Feature engineering

# 3. Frequency-based encoding for Country and Platform
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

    