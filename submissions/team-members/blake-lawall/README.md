# Social Media Addiction Analysis

## Project Description
This project analyzes patterns of social media addiction among students, exploring relationships between usage patterns, academic performance, mental health, and demographic factors. The analysis combines various data science techniques including exploratory data analysis, feature engineering, and dimensionality reduction.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Module Descriptions](#module-descriptions)
4. [Key Findings](#key-findings)
5. [Analysis Methods](#analysis-methods)
6. [Usage Examples](#usage-examples)

## Project Structure
```
Social-Sphere1/
├── data/                      # Data files
│   ├── Students Social Media Addiction.csv
│   └── preprocessed_students_data.csv
├── outputs/                   # Generated outputs
│   ├── visualizations/       # General visualizations
│   ├── clustering/           # Clustering analysis results
│   └── feature_analysis/     # Feature engineering results
├── src/                      # Source code
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── data_visualization.py # Visualization scripts
│   ├── clustering_analysis.py# Clustering and dimensionality reduction
│   ├── target_analysis.py    # Target variable analysis
│   └── feature_encoding.py   # Feature engineering
├── README.md                 # Project documentation
└── pyproject.toml            # Project dependencies
```

## Installation

### Prerequisites
- Python 3.12 or higher
- UV package manager

### Setup Steps
1. Clone the repository:
```bash
git clone [repository-url]
cd Social-Sphere1
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r pyproject.toml
```

## Module Descriptions

### 1. Data Preprocessing (`data_preprocessing.py`)
- Handles initial data cleaning and preparation
- Performs missing value checks and handling
- Implements data type conversions
- Conducts basic statistical analysis
- Output: preprocessed_students_data.csv

### 2. Data Visualization (`data_visualization.py`)
- Creates comprehensive visualizations
- Generates distribution plots
- Produces correlation matrices
- Implements demographic analysis plots
- Output: Various visualization files in outputs/visualizations/

### 3. Clustering Analysis (`clustering_analysis.py`)
- Implements PCA and UMAP dimensionality reduction
- Performs cluster analysis
- Generates 2D projections
- Analyzes behavioral segments
- Output: Clustering results in outputs/clustering/

### 4. Target Analysis (`target_analysis.py`)
- Analyzes addiction scores across demographics
- Creates violin and swarm plots
- Detects and handles outliers
- Examines demographic biases
- Output: Target analysis results in outputs/feature_analysis/

### 5. Feature Engineering (`feature_encoding.py`)
- Implements multiple encoding strategies
- Handles categorical variables
- Performs feature scaling
- Tests for normality
- Output: Encoded features in outputs/feature_analysis/

## Key Findings

### Usage Patterns
- Instagram dominates with 35.3% user preference
- Average daily usage varies significantly by platform
- Peak usage hours correlate with academic schedules

### Academic Impact
- 64.3% report negative academic effects
- Strong correlation between usage hours and academic performance
- Platform-specific variations in academic impact

### Mental Health Correlations
- Significant inverse relationship between social media use and sleep
- Higher addiction scores correlate with increased conflicts
- Platform-specific variations in mental health impact

### Demographic Patterns
- Clear regional preferences in platform usage
- Academic level influences usage patterns
- Cultural variations in social media impact

## Analysis Methods

### Feature Engineering
- Label Encoding for ordinal variables
- One-Hot Encoding for nominal variables
- Frequency Encoding for high-cardinality features
- Ordinal Encoding for ordered categories

### Statistical Analysis
- Correlation analysis
- Chi-square tests for independence
- ANOVA for group comparisons
- Non-parametric tests where appropriate

### Machine Learning Techniques
- PCA (57.18% variance explained)
- UMAP for non-linear dimensionality reduction
- Clustering for behavioral segmentation
- Outlier detection using IQR method

## Usage Examples

### Basic Data Processing
```python
from src.data_preprocessing import preprocess_data
processed_df = preprocess_data("data/Students Social Media Addiction.csv")
```

### Generating Visualizations
```python
from src.data_visualization import create_distribution_plots
create_distribution_plots(processed_df, output_dir="outputs/visualizations")
```

### Running Clustering Analysis
```python
from src.clustering_analysis import perform_clustering
clusters = perform_clustering(processed_df, n_components=2)
```

For more detailed examples and usage instructions, refer to the docstrings in each module. 