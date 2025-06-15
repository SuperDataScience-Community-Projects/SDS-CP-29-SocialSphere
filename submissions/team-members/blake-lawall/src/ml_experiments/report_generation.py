"""
Report Generation Module

This module generates comprehensive reports for ML experiments:
- Model performance summaries
- Feature importance analysis
- Cross-validation results
- Visualization summaries
- MLflow experiment tracking results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import mlflow
from datetime import datetime
from typing import Dict, List, Union
import logging
import jinja2
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentReport:
    """Class for generating experiment reports."""
    
    def __init__(self, experiment_name: str, output_dir: str):
        """
        Initialize report generator.

        Args:
            experiment_name: Name of MLflow experiment
            output_dir: Directory to save reports
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        self.template_dir = Path(__file__).parent / 'templates'
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir))
        )
    
    def collect_mlflow_results(self) -> Dict:
        """Collect results from MLflow tracking."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            raise ValueError(f"Experiment {self.experiment_name} not found")
        
        # Get all runs for the experiment
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        results = {
            'experiment_name': self.experiment_name,
            'runs': []
        }
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'start_time': run.info.start_time,
                'status': run.info.status,
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }
            results['runs'].append(run_data)
        
        return results

    def generate_performance_plots(self,
                                 results: Dict) -> Dict[str, str]:
        """
        Generate performance visualization plots.

        Args:
            results: Dictionary of experiment results

        Returns:
            Dictionary of plot filenames
        """
        plots = {}
        
        # Metric comparison plot
        plt.figure(figsize=(12, 6))
        metrics = pd.DataFrame([
            run['metrics'] for run in results['runs']
        ])
        
        if not metrics.empty:
            sns.boxplot(data=metrics)
            plt.title('Metric Distribution Across Runs')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = self.output_dir / 'metric_distribution.png'
            plt.savefig(plot_path)
            plots['metric_distribution'] = str(plot_path)
            plt.close()
        
        # Parameter importance plot
        if len(results['runs']) > 1:
            params = pd.DataFrame([
                run['params'] for run in results['runs']
            ])
            metrics = pd.DataFrame([
                run['metrics'] for run in results['runs']
            ])
            
            if not params.empty and not metrics.empty:
                plt.figure(figsize=(10, 6))
                for metric in metrics.columns:
                    for param in params.columns:
                        if params[param].nunique() > 1:
                            plt.scatter(params[param], metrics[metric],
                                      alpha=0.5, label=metric)
                
                plt.title('Parameter vs Metric Relationships')
                plt.legend()
                plt.tight_layout()
                
                plot_path = self.output_dir / 'parameter_importance.png'
                plt.savefig(plot_path)
                plots['parameter_importance'] = str(plot_path)
                plt.close()
        
        return plots

    def create_summary_tables(self, results: Dict) -> Dict[str, pd.DataFrame]:
        """
        Create summary tables from results.

        Args:
            results: Dictionary of experiment results

        Returns:
            Dictionary of summary DataFrames
        """
        summaries = {}
        
        # Run summary
        run_data = []
        for run in results['runs']:
            run_info = {
                'Run ID': run['run_id'],
                'Status': run['status'],
                'Start Time': datetime.fromtimestamp(
                    run['start_time']/1000
                ).strftime('%Y-%m-%d %H:%M:%S')
            }
            run_info.update(run['metrics'])
            run_data.append(run_info)
        
        if run_data:
            summaries['runs'] = pd.DataFrame(run_data)
        
        # Parameter summary
        param_data = []
        for run in results['runs']:
            param_data.append(run['params'])
        
        if param_data:
            summaries['parameters'] = pd.DataFrame(param_data)
        
        return summaries

    def generate_html_report(self,
                           results: Dict,
                           plots: Dict[str, str],
                           summaries: Dict[str, pd.DataFrame]) -> str:
        """
        Generate HTML report.

        Args:
            results: Experiment results
            plots: Dictionary of plot filenames
            summaries: Dictionary of summary tables

        Returns:
            HTML report content
        """
        template = self.env.get_template('report_template.html')
        
        # Convert plots to base64
        encoded_plots = {}
        for name, path in plots.items():
            with open(path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                encoded_plots[name] = encoded
        
        # Convert DataFrames to HTML
        tables = {
            name: df.to_html(classes='table table-striped')
            for name, df in summaries.items()
        }
        
        # Render template
        html_content = template.render(
            experiment_name=self.experiment_name,
            plots=encoded_plots,
            tables=tables,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content

    def save_report(self, html_content: str):
        """
        Save HTML report to file.

        Args:
            html_content: HTML report content
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"report_{timestamp}.html"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {report_path}")

    def generate_report(self):
        """Generate complete experiment report."""
        # Collect results
        results = self.collect_mlflow_results()
        
        # Generate visualizations
        plots = self.generate_performance_plots(results)
        
        # Create summary tables
        summaries = self.create_summary_tables(results)
        
        # Generate HTML report
        html_content = self.generate_html_report(results, plots, summaries)
        
        # Save report
        self.save_report(html_content)

def main():
    """Main function to demonstrate report generation."""
    # Create sample MLflow experiment
    experiment_name = "sample_experiment"
    mlflow.set_experiment(experiment_name)
    
    # Create some sample runs
    for i in range(3):
        with mlflow.start_run():
            # Log some metrics
            mlflow.log_metric("accuracy", np.random.rand())
            mlflow.log_metric("f1_score", np.random.rand())
            
            # Log some parameters
            mlflow.log_param("n_estimators", 100 * (i + 1))
            mlflow.log_param("max_depth", 5 * (i + 1))
    
    # Generate report
    output_dir = 'results/reports'
    report_generator = ExperimentReport(experiment_name, output_dir)
    report_generator.generate_report()

if __name__ == "__main__":
    main() 