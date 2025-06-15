"""
Feature Engineering Module for Social Media Addiction Analysis

This module handles the encoding and transformation of features for the social media
addiction analysis. It implements various encoding strategies for categorical variables
and provides functionality for feature scaling and normality testing.

Functions:
    - encode_categorical: Apply various encoding strategies to categorical variables
    - perform_feature_scaling: Scale numerical features
    - test_normality: Test for normal distribution in features
    - create_interaction_features: Create interaction terms between features
    - analyze_feature_importance: Analyze importance of engineered features

Usage:
    from feature_encoding import encode_features
    encoded_df = encode_features(df, encoding_strategy='all')
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')

# Set the style for all plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Create output directory if it doesn't exist
Path("outputs/feature_analysis").mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the dataset"""
    return pd.read_csv('data/Students Social Media Addiction.csv')

def analyze_feature_distributions(df, output_dir):
    """Analyze and plot distributions of numerical features"""
    print("\nAnalyzing Feature Distributions...")
    
    numerical_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                     'Mental_Health_Score', 'Addicted_Score', 'Conflicts_Over_Social_Media']
    
    # Create distribution plots with normality tests
    for col in numerical_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original distribution
        sns.histplot(data=df, x=col, kde=True, ax=ax1)
        ax1.set_title(f'Original Distribution of {col}')
        
        # Q-Q plot
        stats.probplot(df[col], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        # Perform Shapiro-Wilk test
        stat, p_value = stats.shapiro(df[col])
        
        plt.suptitle(f'Distribution Analysis of {col}\nShapiro-Wilk p-value: {p_value:.4f}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_{col}.png')
        plt.close()
        
        # If data is significantly non-normal, try log transformation
        if p_value < 0.05 and (df[col] > 0).all():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            log_data = np.log1p(df[col])
            
            # Log-transformed distribution
            sns.histplot(data=log_data, kde=True, ax=ax1)
            ax1.set_title(f'Log-Transformed Distribution of {col}')
            
            # Q-Q plot for log-transformed data
            stats.probplot(log_data, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Log-Transformed)')
            
            # Perform Shapiro-Wilk test on log-transformed data
            stat_log, p_value_log = stats.shapiro(log_data)
            
            plt.suptitle(f'Log-Transformed Analysis of {col}\nShapiro-Wilk p-value: {p_value_log:.4f}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/distribution_{col}_log.png')
            plt.close()
            
            print(f"\n{col}:")
            print(f"Original Shapiro-Wilk p-value: {p_value:.4f}")
            print(f"Log-transformed Shapiro-Wilk p-value: {p_value_log:.4f}")

def encode_categorical(df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      strategy: str = 'all') -> Tuple[pd.DataFrame, Dict]:
    """
    Apply various encoding strategies to categorical variables.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (Optional[List[str]]): Specific columns to encode. If None, all
            categorical columns will be encoded.
        strategy (str): Encoding strategy ('label', 'onehot', 'frequency', 'all')

    Returns:
        Tuple[pd.DataFrame, Dict]: Encoded dataframe and encoding information

    Example:
        >>> encoded_df, encoders = encode_categorical(df, strategy='all')
        >>> print(encoded_df.shape)
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    if strategy in ['label', 'all']:
        # Label Encoding
        label_encoders = {}
        for col in columns:
            le = LabelEncoder()
            df_encoded[f'{col}_label'] = le.fit_transform(df[col])
            label_encoders[col] = le
        encoding_info['label'] = label_encoders
    
    if strategy in ['onehot', 'all']:
        # One-Hot Encoding
        for col in columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        encoding_info['onehot'] = columns
    
    if strategy in ['frequency', 'all']:
        # Frequency Encoding
        freq_encodings = {}
        for col in columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq'] = df[col].map(freq_encoding)
            freq_encodings[col] = freq_encoding
        encoding_info['frequency'] = freq_encodings
    
    return df_encoded, encoding_info

def perform_feature_scaling(df: pd.DataFrame,
                          columns: Optional[List[str]] = None,
                          scaler_type: str = 'standard') -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using specified scaling method.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (Optional[List[str]]): Columns to scale. If None, all numerical
            columns will be scaled.
        scaler_type (str): Type of scaling to perform ('standard', 'minmax', 'robust')

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Scaled dataframe and fitted scaler

    Example:
        >>> scaled_df, scaler = perform_feature_scaling(df)
        >>> print(scaled_df.describe())
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    scaler = StandardScaler()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler

def test_normality(df: pd.DataFrame,
                  columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Test for normal distribution in features.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (Optional[List[str]]): Columns to test. If None, all numerical
            columns will be tested.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing normality test results

    Example:
        >>> normality_results = test_normality(df)
        >>> print(normality_results['Age']['p_value'])
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    results = {}
    for col in columns:
        statistic, p_value = stats.normaltest(df[col])
        results[col] = {
            'statistic': statistic,
            'p_value': p_value
        }
    
    return results

def create_interaction_features(df: pd.DataFrame,
                              feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction terms between specified feature pairs.

    Args:
        df (pd.DataFrame): Input dataframe
        feature_pairs (List[Tuple[str, str]]): List of feature pairs to interact

    Returns:
        pd.DataFrame: Dataframe with added interaction features

    Example:
        >>> pairs = [('Age', 'Daily_Usage_Hours')]
        >>> df_with_interactions = create_interaction_features(df, pairs)
    """
    df_interactions = df.copy()
    
    for feat1, feat2 in feature_pairs:
        interaction_name = f"{feat1}_{feat2}_interaction"
        df_interactions[interaction_name] = df[feat1] * df[feat2]
    
    return df_interactions

def analyze_feature_importance(df: pd.DataFrame,
                             target: str,
                             output_dir: str) -> Dict[str, float]:
    """
    Analyze importance of engineered features.

    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target variable name
        output_dir (str): Directory to save analysis plots

    Returns:
        Dict[str, float]: Dictionary containing feature importance scores

    Example:
        >>> importance = analyze_feature_importance(df, 'Addiction_Score',
                                                  'outputs/feature_analysis')
        >>> print(importance)
    """
    # Create output directory
    feature_dir = Path(output_dir) / "feature_importance"
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate correlations with target
    correlations = df.corr()[target].sort_values(ascending=False)
    
    # Create correlation plot
    plt.figure(figsize=(12, 8))
    correlations.plot(kind='bar')
    plt.title(f'Feature Correlations with {target}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(feature_dir / "feature_correlations.png")
    plt.close()
    
    return correlations.to_dict()

def main():
    """
    Main function to execute feature engineering pipeline.
    """
    # Load data
    df = load_data()
    
    # Analyze feature distributions
    analyze_feature_distributions(df, 'outputs/feature_analysis')
    
    # Encode categorical variables
    df_encoded, encoding_info = encode_categorical(df)
    
    # Scale features
    df_scaled, scaler = perform_feature_scaling(df_encoded)
    
    # Test for normality
    normality_results = test_normality(df_scaled)
    print("\nNormality Test Results:")
    for feature, results in normality_results.items():
        print(f"\n{feature}:")
        print(f"p-value: {results['p_value']:.4f}")
    
    # Create interaction features
    feature_pairs = [
        ('Age', 'Daily_Usage_Hours'),
        ('Sleep_Hours', 'Daily_Usage_Hours')
    ]
    df_with_interactions = create_interaction_features(df_scaled, feature_pairs)
    
    # Analyze feature importance
    importance_scores = analyze_feature_importance(
        df_with_interactions, 'Addiction_Score', str(Path('outputs/feature_analysis'))
    )
    
    print("\nTop 5 Most Important Features:")
    top_features = dict(sorted(importance_scores.items(), 
                             key=lambda x: abs(x[1]), 
                             reverse=True)[:5])
    for feature, score in top_features.items():
        print(f"{feature}: {score:.3f}")
    
    print("\nFeature engineering completed successfully.")

if __name__ == "__main__":
    main() 