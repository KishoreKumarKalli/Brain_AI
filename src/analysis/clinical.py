"""
Clinical data analysis module for brain segmentation framework.
This module provides functionality for statistical analysis of clinical data
and correlation with imaging findings.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import pingouin as pg
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('clinical_analysis')


class ClinicalAnalyzer:
    """
    Class for analyzing clinical data and correlating with imaging findings.
    """

    def __init__(self, output_dir="./analysis_results"):
        """
        Initialize the clinical data analyzer.

        Args:
            output_dir (str): Directory to save analysis results
        """
        # Metadata
        self.analysis_date = "2025-04-02 14:51:46"
        self.analyst = "KishoreKumarKalli"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results storage
        self.results = {}

        logger.info(f"Clinical analyzer initialized at {self.analysis_date} by {self.analyst}")
        logger.info(f"Results will be saved to: {output_dir}")

    def load_clinical_data(self, filepath):
        """
        Load clinical data from a CSV file.

        Args:
            filepath (str): Path to the clinical data CSV file

        Returns:
            pandas.DataFrame: Loaded clinical data
        """
        try:
            logger.info(f"Loading clinical data from {filepath}")
            data = pd.read_csv(filepath)
            logger.info(f"Successfully loaded clinical data with {data.shape[0]} rows and {data.shape[1]} columns")

            # Display the first few rows and column info
            logger.info(f"Clinical data columns: {data.columns.tolist()}")
            logger.info(f"Clinical data preview:\n{data.head()}")

            return data
        except Exception as e:
            logger.error(f"Error loading clinical data: {str(e)}")
            raise

    def describe_variables(self, data, save_to_file=True):
        """
        Generate descriptive statistics for clinical variables.

        Args:
            data (pandas.DataFrame): Clinical data
            save_to_file (bool): Whether to save results to a file

        Returns:
            dict: Dictionary with descriptive statistics
        """
        logger.info("Generating descriptive statistics for clinical variables")

        # Separate numeric and categorical variables
        numeric_vars = data.select_dtypes(include=['number']).columns.tolist()
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()

        logger.info(f"Numeric variables: {numeric_vars}")
        logger.info(f"Categorical variables: {categorical_vars}")

        # Descriptive statistics for numeric variables
        numeric_stats = data[numeric_vars].describe().transpose()
        numeric_stats['missing'] = data[numeric_vars].isnull().sum()
        numeric_stats['missing_percent'] = (data[numeric_vars].isnull().sum() / len(data)) * 100

        # Frequency tables for categorical variables
        categorical_stats = {}
        for var in categorical_vars:
            value_counts = data[var].value_counts(dropna=False)
            proportions = data[var].value_counts(dropna=False, normalize=True) * 100
            categorical_stats[var] = pd.DataFrame({
                'count': value_counts,
                'percentage': proportions
            })

        # Store results
        results = {
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats
        }

        # Save results to file if requested
        if save_to_file:
            # Save numeric statistics
            numeric_stats_path = os.path.join(self.output_dir, 'numeric_descriptive_stats.csv')
            numeric_stats.to_csv(numeric_stats_path)
            logger.info(f"Saved numeric statistics to {numeric_stats_path}")

            # Save categorical statistics
            for var, stats_df in categorical_stats.items():
                cat_stats_path = os.path.join(self.output_dir, f'categorical_stats_{var}.csv')
                stats_df.to_csv(cat_stats_path)
                logger.info(f"Saved categorical statistics for {var} to {cat_stats_path}")

        self.results['descriptive_statistics'] = results
        logger.info("Descriptive statistics generation completed")

        return results

    def analyze_group_differences(self, data, group_var, outcome_vars, save_to_file=True):
        """
        Analyze differences between groups on clinical variables.

        Args:
            data (pandas.DataFrame): Clinical data
            group_var (str): Name of the grouping variable (e.g., 'DiagnosticGroup')
            outcome_vars (list): List of outcome variables to analyze
            save_to_file (bool): Whether to save results to a file

        Returns:
            dict: Dictionary with analysis results
        """
        logger.info(f"Analyzing group differences for {len(outcome_vars)} variables by {group_var}")

        results = {}

        # Check if the grouping variable exists
        if group_var not in data.columns:
            logger.error(f"Grouping variable {group_var} not found in the data")
            return results

        # Ensure outcome variables exist in the data
        valid_vars = [var for var in outcome_vars if var in data.columns]
        if len(valid_vars) != len(outcome_vars):
            missing_vars = set(outcome_vars) - set(valid_vars)
            logger.warning(f"Some outcome variables not found in the data: {missing_vars}")

        # Get the groups
        groups = data[group_var].unique()
        logger.info(f"Groups in analysis: {groups}")

        # Analyze each outcome variable
        for var in valid_vars:
            logger.info(f"Analyzing {var} by {group_var}")

            # Create a result dictionary for this variable
            var_result = {}

            # Check if the variable is numeric
            if pd.api.types.is_numeric_dtype(data[var]):
                # Descriptive statistics by group
                group_stats = data.groupby(group_var)[var].describe()
                var_result['descriptive_stats'] = group_stats

                # Check for normality using Shapiro-Wilk test
                normality_tests = {}
                for group in groups:
                    group_data = data[data[group_var] == group][var].dropna()
                    if len(group_data) < 3:
                        normality_tests[group] = {'W': None, 'p-value': None, 'normal': None}
                        continue

                    if len(group_data) <= 5000:  # Shapiro-Wilk limited to 5000 samples
                        w, p = stats.shapiro(group_data)
                        normality_tests[group] = {
                            'W': w,
                            'p-value': p,
                            'normal': p > 0.05
                        }
                    else:
                        # For larger samples, use D'Agostino-Pearson test
                        k2, p = stats.normaltest(group_data)
                        normality_tests[group] = {
                            'K²': k2,
                            'p-value': p,
                            'normal': p > 0.05
                        }

                var_result['normality_tests'] = normality_tests

                # Check if all groups are approximately normal
                all_normal = all(test.get('normal', False) for test in normality_tests.values()
                                 if test.get('normal') is not None)

                # Perform appropriate statistical test
                if len(groups) == 2:
                    # Two groups - t-test or Mann-Whitney U test
                    group1_data = data[data[group_var] == groups[0]][var].dropna()
                    group2_data = data[data[group_var] == groups[1]][var].dropna()

                    if all_normal:
                        # Equal variance test
                        lev_stat, lev_p = stats.levene(group1_data, group2_data)
                        var_result['variance_test'] = {'stat': lev_stat, 'p-value': lev_p, 'equal_var': lev_p > 0.05}

                        # t-test
                        equal_var = lev_p > 0.05
                        t_stat, t_p = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                        var_result['ttest'] = {'t': t_stat, 'p-value': t_p, 'significant': t_p < 0.05}
                        var_result['test_used'] = f"t-test (equal_var={equal_var})"
                    else:
                        # Mann-Whitney U test
                        u_stat, u_p = stats.mannwhitneyu(group1_data, group2_data)
                        var_result['mannwhitneyu'] = {'U': u_stat, 'p-value': u_p, 'significant': u_p < 0.05}
                        var_result['test_used'] = "Mann-Whitney U test"

                elif len(groups) > 2:
                    # More than two groups - ANOVA or Kruskal-Wallis
                    if all_normal:
                        # One-way ANOVA
                        groups_data = [data[data[group_var] == group][var].dropna() for group in groups]

                        # Check for equal variances
                        lev_stat, lev_p = stats.levene(*groups_data)
                        var_result['variance_test'] = {'stat': lev_stat, 'p-value': lev_p, 'equal_var': lev_p > 0.05}

                        # ANOVA
                        f_stat, f_p = stats.f_oneway(*groups_data)
                        var_result['anova'] = {'F': f_stat, 'p-value': f_p, 'significant': f_p < 0.05}
                        var_result['test_used'] = "One-way ANOVA"

                        # If ANOVA is significant, perform post-hoc tests
                        if f_p < 0.05:
                            # Create a DataFrame for post-hoc tests
                            posthoc_data = data[[group_var, var]].dropna()

                            # Tukey's HSD
                            tukey = pairwise_tukeyhsd(posthoc_data[var], posthoc_data[group_var])
                            tukey_results = pd.DataFrame(data=tukey._results_table.data[1:],
                                                         columns=tukey._results_table.data[0])
                            var_result['posthoc_tukey'] = tukey_results
                    else:
                        # Kruskal-Wallis test
                        groups_data = [data[data[group_var] == group][var].dropna() for group in groups]
                        h_stat, h_p = stats.kruskal(*groups_data)
                        var_result['kruskal'] = {'H': h_stat, 'p-value': h_p, 'significant': h_p < 0.05}
                        var_result['test_used'] = "Kruskal-Wallis test"

                        # If Kruskal-Wallis is significant, perform post-hoc tests
                        if h_p < 0.05:
                            # Dunn's test
                            try:
                                # Create a DataFrame suitable for pingouin's pairwise_tests
                                posthoc_data = data[[group_var, var]].dropna()
                                dunn = pg.pairwise_tests(data=posthoc_data, dv=var, between=group_var,
                                                         parametric=False, p_adjust='bonferroni')
                                var_result['posthoc_dunn'] = dunn
                            except Exception as e:
                                logger.warning(f"Error in Dunn's test: {str(e)}")

            else:
                # Categorical variable - Chi-square test
                contingency_table = pd.crosstab(data[group_var], data[var])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                var_result['chi2_test'] = {
                    'chi2': chi2,
                    'p-value': p,
                    'dof': dof,
                    'significant': p < 0.05
                }
                var_result['contingency_table'] = contingency_table
                var_result['test_used'] = "Chi-square test"

            # Store results for this variable
            results[var] = var_result

        # Save results to file if requested
        if save_to_file:
            result_path = os.path.join(self.output_dir, f'group_differences_{group_var}.txt')
            with open(result_path, 'w') as f:
                f.write(f"Group Differences Analysis for {group_var}\n")
                f.write(f"Generated at: 2025-04-02 14:52:40\n")
                f.write(f"Generated by: KishoreKumarKalli\n\n")

                for var, var_result in results.items():
                    f.write(f"===== {var} =====\n")
                    f.write(f"Test used: {var_result.get('test_used', 'Unknown')}\n\n")

                    if 'descriptive_stats' in var_result:
                        f.write("Descriptive Statistics:\n")
                        f.write(var_result['descriptive_stats'].to_string())
                        f.write("\n\n")

                    if 'ttest' in var_result:
                        test_result = var_result['ttest']
                        f.write(f"t-test: t={test_result['t']:.4f}, p={test_result['p-value']:.4f}\n")
                        f.write(f"Significant: {test_result['significant']}\n\n")

                    if 'mannwhitneyu' in var_result:
                        test_result = var_result['mannwhitneyu']
                        f.write(f"Mann-Whitney U test: U={test_result['U']:.4f}, p={test_result['p-value']:.4f}\n")
                        f.write(f"Significant: {test_result['significant']}\n\n")

                    if 'anova' in var_result:
                        test_result = var_result['anova']
                        f.write(f"ANOVA: F={test_result['F']:.4f}, p={test_result['p-value']:.4f}\n")
                        f.write(f"Significant: {test_result['significant']}\n\n")

                    if 'kruskal' in var_result:
                        test_result = var_result['kruskal']
                        f.write(f"Kruskal-Wallis test: H={test_result['H']:.4f}, p={test_result['p-value']:.4f}\n")
                        f.write(f"Significant: {test_result['significant']}\n\n")

                    if 'chi2_test' in var_result:
                        test_result = var_result['chi2_test']
                        f.write(
                            f"Chi-square test: χ²={test_result['chi2']:.4f}, p={test_result['p-value']:.4f}, dof={test_result['dof']}\n")
                        f.write(f"Significant: {test_result['significant']}\n\n")

                        f.write("Contingency Table:\n")
                        f.write(var_result['contingency_table'].to_string())
                        f.write("\n\n")

            logger.info(f"Saved group differences analysis to {result_path}")

        # Store in results dictionary
        self.results['group_differences'] = results
        logger.info("Group differences analysis completed")

        return results

    def correlate_clinical_imaging(self, clinical_data, imaging_data, clinical_vars, imaging_vars, save_to_file=True):
        """
        Correlate clinical variables with imaging variables.

        Args:
            clinical_data (pandas.DataFrame): Clinical data
            imaging_data (pandas.DataFrame): Imaging data with the same subject IDs
            clinical_vars (list): Clinical variables to correlate
            imaging_vars (list): Imaging variables to correlate
            save_to_file (bool): Whether to save results to a file

        Returns:
            pandas.DataFrame: Correlation matrix
        """
        logger.info(f"Correlating {len(clinical_vars)} clinical variables with {len(imaging_vars)} imaging variables")

        # Ensure all variables exist
        for var in clinical_vars:
            if var not in clinical_data.columns:
                logger.warning(f"Clinical variable {var} not found in the data")
                clinical_vars.remove(var)

        for var in imaging_vars:
            if var not in imaging_data.columns:
                logger.warning(f"Imaging variable {var} not found in the data")
                imaging_vars.remove(var)

        # Merge clinical and imaging data
        # Assuming both dataframes have a common ID column
        id_column = None
        for col in clinical_data.columns:
            if 'id' in col.lower() or 'subject' in col.lower():
                if col in imaging_data.columns:
                    id_column = col
                    break

        if id_column is None:
            logger.error("Cannot find common ID column between clinical and imaging data")
            return None

        # Merge the dataframes
        merged_data = pd.merge(clinical_data, imaging_data, on=id_column)
        logger.info(f"Merged data has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns")

        # Calculate correlations for numeric variables
        numeric_clinical_vars = [var for var in clinical_vars
                                 if pd.api.types.is_numeric_dtype(merged_data[var])]
        numeric_imaging_vars = [var for var in imaging_vars
                                if pd.api.types.is_numeric_dtype(merged_data[var])]

        if not numeric_clinical_vars or not numeric_imaging_vars:
            logger.warning("No numeric variables to correlate")
            return None

        # Calculate correlation matrix
        correlation_matrix = pd.DataFrame(index=numeric_clinical_vars, columns=numeric_imaging_vars)
        pvalue_matrix = pd.DataFrame(index=numeric_clinical_vars, columns=numeric_imaging_vars)

        for clin_var in numeric_clinical_vars:
            for img_var in numeric_imaging_vars:
                # Remove rows with missing values
                valid_data = merged_data[[clin_var, img_var]].dropna()

                if len(valid_data) > 2:  # Need at least 3 points for correlation
                    # Calculate Pearson correlation
                    r, p = stats.pearsonr(valid_data[clin_var], valid_data[img_var])
                    correlation_matrix.loc[clin_var, img_var] = r
                    pvalue_matrix.loc[clin_var, img_var] = p

        # Create a combined result with correlation coefficients and p-values
        result = {
            'correlation': correlation_matrix,
            'pvalue': pvalue_matrix
        }

        # Save results to file if requested
        if save_to_file:
            corr_path = os.path.join(self.output_dir, 'clinical_imaging_correlations.csv')
            correlation_matrix.to_csv(corr_path)
            logger.info(f"Saved correlation matrix to {corr_path}")

            pval_path = os.path.join(self.output_dir, 'clinical_imaging_pvalues.csv')
            pvalue_matrix.to_csv(pval_path)
            logger.info(f"Saved p-value matrix to {pval_path}")

            # Create a heatmap visualization
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                        mask=mask, vmin=-1, vmax=1,
                        linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title(f'Clinical-Imaging Correlations\nGenerated: 2025-04-02 14:52:40', fontsize=14)
            plt.tight_layout()

            heatmap_path = os.path.join(self.output_dir, 'clinical_imaging_correlation_heatmap.png')
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            logger.info(f"Saved correlation heatmap to {heatmap_path}")

        # Store in results dictionary
        self.results['clinical_imaging_correlation'] = result
        logger.info("Clinical-imaging correlation analysis completed")

        return result

    def cluster_analysis(self, data, variables, n_clusters=3, method='kmeans', save_to_file=True):
        """
        Perform cluster analysis on clinical data.

        Args:
            data (pandas.DataFrame): Clinical data
            variables (list): Variables to use for clustering
            n_clusters (int): Number of clusters to find
            method (str): Clustering method ('kmeans' or 'hierarchical')
            save_to_file (bool): Whether to save results to a file

        Returns:
            dict: Clustering results
        """
        logger.info(f"Performing {method} clustering with {n_clusters} clusters using {len(variables)} variables")

        # Check if variables exist
        valid_vars = [var for var in variables if var in data.columns]
        if len(valid_vars) != len(variables):
            missing_vars = set(variables) - set(valid_vars)
            logger.warning(f"Some variables not found in the data: {missing_vars}")

        # Prepare the data for clustering
        cluster_data = data[valid_vars].copy()

        # Handle missing values
        if cluster_data.isnull().any().any():
            logger.warning("Data contains missing values. Rows with missing values will be removed.")
            cluster_data = cluster_data.dropna()

        if len(cluster_data) < n_clusters:
            logger.error(f"Not enough data points ({len(cluster_data)}) for {n_clusters} clusters")
            return None

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Perform dimensionality reduction for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Perform clustering
        if method.lower() == 'kmeans':
            # K-means clustering
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(scaled_data)
            centers = model.cluster_centers_

            # Transform cluster centers back to original space
            original_centers = scaler.inverse_transform(centers)

            # Calculate silhouette score
            silhouette_avg = silhouette_score(scaled_data, labels)

            # Calculate cluster sizes
            cluster_sizes = np.bincount(labels)

            # Prepare results
            result = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'labels': labels,
                'cluster_centers': original_centers,
                'silhouette_score': silhouette_avg,
                'cluster_sizes': cluster_sizes,
                'pca_result': pca_result,
                'pca_explained_variance_ratio': pca.explained_variance_ratio_
            }

        else:
            # Default to K-means if method is not recognized
            logger.warning(f"Clustering method '{method}' not recognized. Using K-means instead.")
            return self.cluster_analysis(data, variables, n_clusters, 'kmeans', save_to_file)

        # Save results to file if requested
        if save_to_file:
            # Save cluster assignments
            cluster_assignments = data.copy()
            cluster_assignments['cluster'] = pd.Series(labels, index=cluster_data.index)

            assignments_path = os.path.join(self.output_dir, f'{method}_clusters_{n_clusters}.csv')
            cluster_assignments.to_csv(assignments_path, index=False)
            logger.info(f"Saved cluster assignments to {assignments_path}")

            # Create a scatter plot of the clusters (using PCA for 2D visualization)
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'Cluster Analysis ({method}, {n_clusters} clusters)\nGenerated: 2025-04-02 14:52:40')
            plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

            scatter_path = os.path.join(self.output_dir, f'{method}_clusters_{n_clusters}_scatter.png')
            plt.savefig(scatter_path, dpi=300)
            plt.close()
            logger.info(f"Saved cluster scatter plot to {scatter_path}")

        # Store in results dictionary
        self.results['cluster_analysis'] = result
        logger.info("Cluster analysis completed")

        return result