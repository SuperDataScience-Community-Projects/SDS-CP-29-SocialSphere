"""
Target Analysis Module for Social Media Addiction Analysis

This module focuses on analyzing the target variables (addiction scores and conflicts)
across different demographics and factors. It includes functions for creating detailed
visualizations and statistical analyses of addiction patterns.

Functions:
    - prepare_target_variables: Prepare and transform target variables
    - analyze_addiction_by_demographic: Analyze addiction patterns across demographics
    - detect_outliers: Identify outliers in addiction scores
    - create_violin_plots: Generate violin plots for addiction analysis
    - analyze_demographic_bias: Examine demographic biases in addiction scores

Usage:
    from target_analysis import analyze_addiction_patterns
    results = analyze_addiction_patterns(df, target_var='Addiction_Score')
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Set plotting styles
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

def prepare_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and transform target variables for analysis.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with prepared target variables

    Example:
        >>> df_prepared = prepare_target_variables(df)
        >>> print(df_prepared['Addiction_Level'].value_counts())
    """
    df_prepared = df.copy()
    
    # Create addiction level categories
    df_prepared['Addiction_Level'] = pd.qcut(
        df_prepared['Addiction_Score'],
        q=4,
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Convert conflicts to binary
    df_prepared['High_Conflict'] = (df_prepared['Conflicts_Over_Social_Media'] >= 3).astype(int)
    
    return df_prepared

def analyze_addiction_by_demographic(df: pd.DataFrame,
                                  demographic_var: str,
                                  output_dir: str) -> Dict[str, float]:
    """
    Analyze addiction patterns across demographic groups.

    Args:
        df (pd.DataFrame): Input dataframe
        demographic_var (str): Demographic variable to analyze
        output_dir (str): Directory to save analysis plots

    Returns:
        Dict[str, float]: Dictionary containing statistical results

    Example:
        >>> stats = analyze_addiction_by_demographic(df, 'Gender', 'outputs/target_analysis')
        >>> print(stats['p_value'])
    """
    # Create output directory
    demo_dir = Path(output_dir) / "demographics"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform statistical test
    groups = [group for _, group in df.groupby(demographic_var)['Addiction_Score']]
    f_stat, p_value = stats.f_oneway(*groups)
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x=demographic_var, y='Addiction_Score')
    plt.title(f'Addiction Score Distribution by {demographic_var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(demo_dir / f"addiction_by_{demographic_var.lower()}.png")
    plt.close()
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value
    }

def detect_outliers(df: pd.DataFrame,
                   columns: List[str],
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Identify outliers using the IQR method.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str]): Columns to check for outliers
        threshold (float): IQR multiplier for outlier detection

    Returns:
        pd.DataFrame: Dataframe containing outlier information

    Example:
        >>> outliers = detect_outliers(df, ['Addiction_Score'], threshold=1.5)
        >>> print(f"Number of outliers: {len(outliers)}")
    """
    outliers_df = pd.DataFrame()
    
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_df = pd.concat([outliers_df, outliers])
    
    return outliers_df.drop_duplicates()

def create_violin_plots(df: pd.DataFrame,
                       target_var: str,
                       group_vars: List[str],
                       output_dir: str) -> None:
    """
    Generate violin plots for analyzing target variable distributions.

    Args:
        df (pd.DataFrame): Input dataframe
        target_var (str): Target variable to analyze
        group_vars (List[str]): Variables to group by
        output_dir (str): Directory to save plots

    Example:
        >>> create_violin_plots(df, 'Addiction_Score', 
                              ['Gender', 'Academic_Level'], 
                              'outputs/target_analysis')
    """
    # Create output directory
    violin_dir = Path(output_dir) / "violin_plots"
    violin_dir.mkdir(parents=True, exist_ok=True)
    
    for group_var in group_vars:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x=group_var, y=target_var)
        plt.title(f'{target_var} Distribution by {group_var}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(violin_dir / f"{target_var.lower()}_by_{group_var.lower()}.png")
        plt.close()

def analyze_demographic_bias(df: pd.DataFrame,
                           demographic_vars: List[str],
                           target_var: str) -> Dict[str, Dict[str, float]]:
    """
    Examine demographic biases in target variable.

    Args:
        df (pd.DataFrame): Input dataframe
        demographic_vars (List[str]): Demographic variables to analyze
        target_var (str): Target variable to analyze

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing bias statistics

    Example:
        >>> bias_stats = analyze_demographic_bias(df, 
                                                ['Gender', 'Academic_Level'],
                                                'Addiction_Score')
        >>> print(bias_stats['Gender']['effect_size'])
    """
    bias_stats = {}
    
    for demo_var in demographic_vars:
        # Calculate mean differences
        group_means = df.groupby(demo_var)[target_var].mean()
        overall_mean = df[target_var].mean()
        max_diff = (group_means - overall_mean).abs().max()
        
        # Calculate effect size (Cohen's d)
        groups = [group for _, group in df.groupby(demo_var)[target_var]]
        pooled_std = np.sqrt(np.mean([np.var(g) for g in groups]))
        effect_size = max_diff / pooled_std
        
        bias_stats[demo_var] = {
            'max_difference': max_diff,
            'effect_size': effect_size
        }
    
    return bias_stats

def main():
    """
    Main function to execute target variable analysis pipeline.
    """
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_students_data.csv')
    
    # Create output directory
    output_dir = Path('outputs/target_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare target variables
    df_prepared = prepare_target_variables(df)
    
    # Analyze addiction by demographics
    demographic_vars = ['Gender', 'Academic_Level', 'Preferred_Social_Media']
    for demo_var in demographic_vars:
        stats = analyze_addiction_by_demographic(df_prepared, demo_var, str(output_dir))
        print(f"\nAnalysis for {demo_var}:")
        print(f"F-statistic: {stats['f_statistic']:.2f}")
        print(f"p-value: {stats['p_value']:.4f}")
    
    # Detect outliers
    outliers = detect_outliers(df_prepared, ['Addiction_Score'])
    print(f"\nNumber of outliers detected: {len(outliers)}")
    
    # Create violin plots
    create_violin_plots(df_prepared, 'Addiction_Score', demographic_vars, str(output_dir))
    
    # Analyze demographic bias
    bias_stats = analyze_demographic_bias(df_prepared, demographic_vars, 'Addiction_Score')
    print("\nDemographic Bias Analysis:")
    for demo_var, stats in bias_stats.items():
        print(f"\n{demo_var}:")
        print(f"Max difference from mean: {stats['max_difference']:.2f}")
        print(f"Effect size (Cohen's d): {stats['effect_size']:.2f}")
    
    print("\nTarget analysis completed successfully.")

if __name__ == "__main__":
    main() 