"""
Clustering Analysis Module for Social Media Addiction Analysis

This module implements dimensionality reduction and clustering techniques to analyze
patterns in social media addiction data. It includes PCA, UMAP, and clustering algorithms
to identify natural groupings in student behavior.

Functions:
    - prepare_data: Prepare data for clustering analysis
    - perform_pca: Implement PCA dimensionality reduction
    - perform_umap: Implement UMAP dimensionality reduction
    - analyze_clusters: Analyze and visualize cluster characteristics
    - plot_projections: Create 2D projection plots with various color codings

Usage:
    from clustering_analysis import perform_clustering_analysis
    clusters = perform_clustering_analysis(df, n_components=2)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import umap
import warnings
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')

# Set the style for all plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Create output directory if it doesn't exist
Path("outputs/clustering").mkdir(parents=True, exist_ok=True)

def prepare_data(df: pd.DataFrame, 
                numerical_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepare data for clustering analysis by scaling numerical features.

    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (Optional[List[str]]): List of numerical columns to use.
            If None, all numerical columns will be used.

    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled data and the fitted scaler

    Example:
        >>> scaled_data, scaler = prepare_data(df, ['Age', 'Daily_Usage_Hours'])
        >>> print(scaled_data.shape)
        (705, 2)
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    return scaled_data, scaler

def perform_pca(scaled_data: np.ndarray, 
                n_components: int = 2) -> Tuple[np.ndarray, PCA, float]:
    """
    Perform PCA dimensionality reduction.

    Args:
        scaled_data (np.ndarray): Standardized input data
        n_components (int): Number of PCA components to compute

    Returns:
        Tuple[np.ndarray, PCA, float]: PCA transformed data, fitted PCA object,
            and explained variance ratio

    Example:
        >>> pca_data, pca, var_ratio = perform_pca(scaled_data)
        >>> print(f"Explained variance ratio: {var_ratio:.2%}")
        Explained variance ratio: 57.18%
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    return pca_data, pca, explained_variance

def perform_umap(scaled_data: np.ndarray,
                n_neighbors: int = 15,
                min_dist: float = 0.1,
                n_components: int = 2) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction.

    Args:
        scaled_data (np.ndarray): Standardized input data
        n_neighbors (int): Number of neighbors for UMAP
        min_dist (float): Minimum distance parameter for UMAP
        n_components (int): Number of components for the embedding

    Returns:
        np.ndarray: UMAP transformed data

    Example:
        >>> umap_data = perform_umap(scaled_data)
        >>> print(umap_data.shape)
        (705, 2)
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       n_components=n_components,
                       random_state=42)
    
    umap_data = reducer.fit_transform(scaled_data)
    return umap_data

def analyze_clusters(df: pd.DataFrame,
                    cluster_labels: np.ndarray,
                    output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Analyze and visualize cluster characteristics.

    Args:
        df (pd.DataFrame): Original dataframe
        cluster_labels (np.ndarray): Cluster assignments
        output_dir (str): Directory to save cluster analysis plots

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing cluster statistics

    Example:
        >>> cluster_stats = analyze_clusters(df, labels, "outputs/clustering")
        >>> print(cluster_stats['cluster_sizes'])
    """
    # Create output directory
    cluster_dir = Path(output_dir)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Analyze cluster sizes
    cluster_sizes = df_with_clusters['Cluster'].value_counts()
    
    # Analyze cluster characteristics
    cluster_stats = df_with_clusters.groupby('Cluster').mean()
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar')
    plt.title('Cluster Sizes')
    plt.savefig(cluster_dir / "cluster_sizes.png")
    plt.close()
    
    return {
        'cluster_sizes': cluster_sizes,
        'cluster_stats': cluster_stats
    }

def plot_projections(projection_data: np.ndarray,
                    df: pd.DataFrame,
                    color_by: str,
                    output_path: str,
                    title: str) -> None:
    """
    Create 2D projection plots with various color codings.

    Args:
        projection_data (np.ndarray): 2D projection data (PCA or UMAP)
        df (pd.DataFrame): Original dataframe
        color_by (str): Column name to use for color coding
        output_path (str): Path to save the plot
        title (str): Plot title

    Example:
        >>> plot_projections(pca_data, df, 'Addiction_Score', 
                           'outputs/clustering/pca_addiction.png',
                           'PCA Projection by Addiction Score')
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projection_data[:, 0], 
                         projection_data[:, 1],
                         c=df[color_by],
                         cmap='viridis')
    plt.colorbar(scatter, label=color_by)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to execute clustering analysis pipeline.
    """
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_students_data.csv')
    
    # Create output directory
    output_dir = Path('outputs/clustering')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    scaled_data, scaler = prepare_data(df)
    
    # Perform PCA
    pca_data, pca, variance_explained = perform_pca(scaled_data)
    print(f"PCA explained variance ratio: {variance_explained:.2%}")
    
    # Perform UMAP
    umap_data = perform_umap(scaled_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(df, cluster_labels, str(output_dir))
    
    # Create projection plots
    plot_projections(pca_data, df, 'Addiction_Score',
                    str(output_dir / 'pca_addiction.png'),
                    'PCA Projection by Addiction Score')
    
    plot_projections(umap_data, df, 'Addiction_Score',
                    str(output_dir / 'umap_addiction.png'),
                    'UMAP Projection by Addiction Score')
    
    print("Clustering analysis completed successfully.")

if __name__ == "__main__":
    main() 