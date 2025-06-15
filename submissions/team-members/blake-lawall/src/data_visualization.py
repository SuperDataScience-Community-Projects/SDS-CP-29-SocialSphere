"""
Data Visualization Module for Social Media Addiction Analysis

This module provides comprehensive visualization functions for analyzing social media addiction patterns.
It includes functions for creating distribution plots, correlation matrices, demographic analyses,
and platform-specific usage visualizations.

Functions:
    - create_distribution_plots: Generate distribution plots for numerical variables
    - plot_correlation_matrix: Create correlation heatmap for numerical variables
    - create_demographic_plots: Generate visualizations for demographic analyses
    - plot_platform_usage: Visualize social media platform usage patterns
    - create_addiction_score_plots: Analyze addiction scores across different factors

Usage:
    from data_visualization import create_distribution_plots, plot_correlation_matrix
    create_distribution_plots(df, output_dir="outputs/visualizations")
    plot_correlation_matrix(df, output_path="outputs/visualizations/correlation_matrix.png")
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set up plotting styles
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

def create_distribution_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate distribution plots for numerical variables.

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save the plots

    Example:
        >>> create_distribution_plots(df, "outputs/visualizations")
        Creating distribution plots...
        Plots saved in outputs/visualizations/distributions/
    """
    # Create output directory
    dist_dir = Path(output_dir) / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        
        # Save plot
        plt.savefig(dist_dir / f"{col.lower()}_distribution.png")
        plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: str) -> None:
    """
    Create correlation heatmap for numerical variables.

    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path to save the correlation matrix plot

    Example:
        >>> plot_correlation_matrix(df, "outputs/visualizations/correlation_matrix.png")
        Correlation matrix saved to outputs/visualizations/correlation_matrix.png
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    # Save plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def create_demographic_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate visualizations for demographic analyses.

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save the demographic plots

    Example:
        >>> create_demographic_plots(df, "outputs/visualizations")
        Creating demographic plots...
        Plots saved in outputs/visualizations/demographics/
    """
    # Create output directory
    demo_dir = Path(output_dir) / "demographics"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', bins=20)
    plt.title('Age Distribution')
    plt.savefig(demo_dir / "age_distribution.png")
    plt.close()
    
    # Gender distribution
    plt.figure(figsize=(8, 6))
    df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Gender Distribution')
    plt.savefig(demo_dir / "gender_distribution.png")
    plt.close()

def plot_platform_usage(df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize social media platform usage patterns.

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save the platform usage plots

    Example:
        >>> plot_platform_usage(df, "outputs/visualizations")
        Creating platform usage plots...
        Plots saved in outputs/visualizations/platform_usage/
    """
    # Create output directory
    platform_dir = Path(output_dir) / "platform_usage"
    platform_dir.mkdir(parents=True, exist_ok=True)
    
    # Platform preference distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Preferred_Social_Media')
    plt.xticks(rotation=45)
    plt.title('Social Media Platform Preferences')
    plt.savefig(platform_dir / "platform_preferences.png", bbox_inches='tight')
    plt.close()
    
    # Usage hours by platform
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Preferred_Social_Media', y='Daily_Usage_Hours')
    plt.xticks(rotation=45)
    plt.title('Daily Usage Hours by Platform')
    plt.savefig(platform_dir / "usage_hours_by_platform.png", bbox_inches='tight')
    plt.close()

def create_addiction_score_plots(df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze addiction scores across different factors.

    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save the addiction score plots

    Example:
        >>> create_addiction_score_plots(df, "outputs/visualizations")
        Creating addiction score plots...
        Plots saved in outputs/visualizations/addiction_scores/
    """
    # Create output directory
    addiction_dir = Path(output_dir) / "addiction_scores"
    addiction_dir.mkdir(parents=True, exist_ok=True)
    
    # Addiction score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Addiction_Score', kde=True)
    plt.title('Distribution of Addiction Scores')
    plt.savefig(addiction_dir / "addiction_score_distribution.png")
    plt.close()
    
    # Addiction score by academic performance
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Academic_Performance', y='Addiction_Score')
    plt.title('Addiction Scores by Academic Performance')
    plt.savefig(addiction_dir / "addiction_by_academic.png")
    plt.close()

def main():
    """
    Main function to execute all visualization analyses.
    """
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_students_data.csv')
    
    # Create base output directory
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    create_distribution_plots(df, str(output_dir))
    plot_correlation_matrix(df, str(output_dir / "correlation_matrix.png"))
    create_demographic_plots(df, str(output_dir))
    plot_platform_usage(df, str(output_dir))
    create_addiction_score_plots(df, str(output_dir))
    
    print("All visualizations have been generated successfully.")

if __name__ == "__main__":
    main() 