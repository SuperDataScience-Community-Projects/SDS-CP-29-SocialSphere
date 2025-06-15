"""
Data Preprocessing Module for Social Media Addiction Analysis

This module handles the initial data cleaning and preparation for the social media addiction analysis project.
It includes functions for loading data, handling missing values, type conversions, and basic statistical analysis.

Functions:
    - load_data: Load the raw dataset from CSV
    - check_missing_values: Analyze and report missing values
    - convert_datatypes: Convert columns to appropriate data types
    - compute_basic_stats: Calculate basic statistical measures
    - preprocess_data: Main function that orchestrates the preprocessing pipeline

Usage:
    from data_preprocessing import preprocess_data
    processed_df = preprocess_data("path/to/data.csv")
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up matplotlib style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_theme()

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw dataset from CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Raw dataframe
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    return pd.read_csv(file_path)

def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze and report missing values in the dataset.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        Dict[str, float]: Dictionary with column names and their missing value percentages
    """
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    return missing_percentages.to_dict()

def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with converted data types
    """
    # Add type conversion logic here
    return df

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate basic statistical measures for numerical columns.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        Dict[str, Dict]: Dictionary containing basic statistics for each numerical column
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    stats = {}
    for col in numerical_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return stats

def preprocess_data(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main function that orchestrates the preprocessing pipeline.

    Args:
        file_path (str): Path to the input CSV file
        output_path (Optional[str]): Path to save the preprocessed data. If None, data won't be saved.

    Returns:
        pd.DataFrame: Preprocessed dataframe

    Example:
        >>> df = preprocess_data("raw_data.csv", "preprocessed_data.csv")
        >>> print(df.shape)
        (705, 20)
    """
    # Load data
    df = load_data(file_path)
    
    # Check missing values
    missing_stats = check_missing_values(df)
    print("Missing Value Analysis:")
    for col, pct in missing_stats.items():
        if pct > 0:
            print(f"{col}: {pct:.2f}%")
    
    # Convert data types
    df = convert_datatypes(df)
    
    # Compute basic statistics
    stats = compute_basic_stats(df)
    print("\nBasic Statistics:")
    for col, stat in stats.items():
        print(f"\n{col}:")
        for metric, value in stat.items():
            print(f"  {metric}: {value:.2f}")
    
    # Save preprocessed data if output path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to {output_path}")
    
    return df

def analyze_missing_data(df):
    """
    Analyze and visualize missing data
    """
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    
    print("\n=== Missing Values Analysis ===")
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentages
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Visualize missing values using missingno
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title("Missing Values Matrix")
    plt.tight_layout()
    plt.savefig('outputs/missing_values_matrix.png')
    
    # Correlation of missingness
    plt.figure(figsize=(10, 8))
    msno.heatmap(df)
    plt.title("Missing Values Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('outputs/missing_values_heatmap.png')

def handle_missing_data(df):
    """
    Handle missing values in the dataset
    """
    df_cleaned = df.copy()
    
    # For numerical columns: impute with median
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # For categorical columns: impute with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned

def handle_extreme_values(df):
    """
    Handle extreme values and inconsistencies
    """
    df_cleaned = df.copy()
    
    # Example: Handle Sleep_Hours_Per_Night > 16
    if 'Sleep_Hours_Per_Night' in df_cleaned.columns:
        print("\n=== Extreme Values Analysis ===")
        print(f"Records with sleep hours > 16: {(df_cleaned['Sleep_Hours_Per_Night'] > 16).sum()}")
        # Cap sleep hours at 16
        df_cleaned.loc[df_cleaned['Sleep_Hours_Per_Night'] > 16, 'Sleep_Hours_Per_Night'] = 16
    
    return df_cleaned

def main():
    # Load and analyze the data
    df = load_and_analyze_data('data/Students Social Media Addiction.csv')
    
    # Analyze missing data
    analyze_missing_data(df)
    
    # Handle missing data
    df_cleaned = handle_missing_data(df)
    
    # Handle extreme values
    df_cleaned = handle_extreme_values(df_cleaned)
    
    # Save the preprocessed dataset
    df_cleaned.to_csv('data/preprocessed_students_data.csv', index=False)
    print("\nPreprocessed data saved to 'data/preprocessed_students_data.csv'")

if __name__ == "__main__":
    main() 