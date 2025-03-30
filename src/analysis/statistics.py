# statistics.py

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pingouin as pg
from datetime import datetime


class BrainMRIStatistics:
    """
    Statistical analysis tools for brain MRI data.
    Includes volume analysis, group comparisons, and clinical correlations.

    Args:
        alpha (float): Significance level for statistical tests (default: 0.05)
        output_dir (str): Directory to save results (default: None)
    """

    def __init__(self, alpha=0.05, output_dir=None):
        self.alpha = alpha
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_volume_statistics(self, segmentation_array, voxel_size=(1.0, 1.0, 1.0)):
        """
        Calculate volume statistics for a segmentation array.

        Args:
            segmentation_array (numpy.ndarray): 3D segmentation mask with different class labels
            voxel_size (tuple): Voxel dimensions in mm (default: 1mm isotropic)

        Returns:
            dict: Dictionary with volume statistics for each class
        """
        # Calculate voxel volume in mm³
        voxel_volume = np.prod(voxel_size)

        # Find unique class labels
        unique_labels = np.unique(segmentation_array)

        # Dictionary to store results
        volumes = {}

        # Calculate statistics for each class label
        for label in unique_labels:
            # Skip background (usually label 0)
            if label == 0:
                continue

            # Count voxels for this class
            voxel_count = np.sum(segmentation_array == label)

            # Calculate volume in mm³
            volume_mm3 = voxel_count * voxel_volume

            # Convert to milliliters for convenience
            volume_ml = volume_mm3 / 1000

            # Store results
            volumes[f'class_{label}_voxel_count'] = voxel_count
            volumes[f'class_{label}_volume_mm3'] = volume_mm3
            volumes[f'class_{label}_volume_ml'] = volume_ml

        # Calculate total brain volume (excluding background)
        total_voxels = np.sum(segmentation_array > 0)
        total_volume_mm3 = total_voxels * voxel_volume
        total_volume_ml = total_volume_mm3 / 1000

        volumes['total_voxel_count'] = total_voxels
        volumes['total_volume_mm3'] = total_volume_mm3
        volumes['total_volume_ml'] = total_volume_ml

        return volumes

    def compare_groups(self, data_df, group_col, measure_cols, paired=False, parametric=True):
        """
        Perform statistical comparison between groups.

        Args:
            data_df (pandas.DataFrame): DataFrame with group and measurement data
            group_col (str): Column name containing group labels
            measure_cols (list): List of column names with measurements to compare
            paired (bool): Whether the data is paired (default: False)
            parametric (bool): Whether to use parametric tests (default: True)

        Returns:
            pandas.DataFrame: DataFrame with statistical test results
        """
        # Initialize results DataFrame
        results = []

        # Get groups
        groups = data_df[group_col].unique()
        n_groups = len(groups)

        for measure in measure_cols:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(data_df[measure]):
                continue

            # For two groups
            if n_groups == 2:
                group1_data = data_df[data_df[group_col] == groups[0]][measure].dropna()
                group2_data = data_df[data_df[group_col] == groups[1]][measure].dropna()

                # Check if we have enough data
                if len(group1_data) < 2 or len(group2_data) < 2:
                    results.append({
                        'measure': measure,
                        'test': 'N/A',
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False,
                        'n_group1': len(group1_data),
                        'n_group2': len(group2_data),
                        'mean_group1': group1_data.mean() if len(group1_data) > 0 else np.nan,
                        'mean_group2': group2_data.mean() if len(group2_data) > 0 else np.nan,
                        'std_group1': group1_data.std() if len(group1_data) > 0 else np.nan,
                        'std_group2': group2_data.std() if len(group2_data) > 0 else np.nan
                    })
                    continue

                # Calculate effect size (Cohen's d)
                mean1, mean2 = group1_data.mean(), group2_data.mean()
                std1, std2 = group1_data.std(), group2_data.std()
                n1, n2 = len(group1_data), len(group2_data)

                # Pooled standard deviation
                pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

                # Cohen's d
                cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan

                if parametric:
                    # Check for normality (simplified)
                    # In a complete implementation, should use proper normality tests
                    use_nonparametric = False

                    # T-test
                    if paired:
                        # Ensure equal length for paired test
                        min_len = min(len(group1_data), len(group2_data))
                        stat, p = stats.ttest_rel(group1_data[:min_len], group2_data[:min_len])
                        test_name = "Paired t-test"
                    elif use_nonparametric:
                        stat, p = stats.mannwhitneyu(group1_data, group2_data)
                        test_name = "Mann-Whitney U"
                    else:
                        stat, p = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        test_name = "Welch's t-test"
                else:
                    # Non-parametric test
                    if paired:
                        min_len = min(len(group1_data), len(group2_data))
                        stat, p = stats.wilcoxon(group1_data[:min_len], group2_data[:min_len])
                        test_name = "Wilcoxon signed-rank"
                    else:
                        stat, p = stats.mannwhitneyu(group1_data, group2_data)
                        test_name = "Mann-Whitney U"

            # For more than two groups
            else:
                group_data = [data_df[data_df[group_col] == g][measure].dropna() for g in groups]
                group_counts = [len(gd) for gd in group_data]
                group_means = [gd.mean() if len(gd) > 0 else np.nan for gd in group_data]
                group_stds = [gd.std() if len(gd) > 0 else np.nan for gd in group_data]

                # Check if we have enough data
                if any(len(gd) < 2 for gd in group_data):
                    result_dict = {
                        'measure': measure,
                        'test': 'N/A',
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False,
                    }

                    # Add group stats to the result
                    for i, group in enumerate(groups):
                        result_dict[f'n_{group}'] = group_counts[i]
                        result_dict[f'mean_{group}'] = group_means[i]
                        result_dict[f'std_{group}'] = group_stds[i]

                    results.append(result_dict)
                    continue

                if parametric:
                    # ANOVA
                    stat, p = stats.f_oneway(*group_data)
                    test_name = "ANOVA"

                    # Post-hoc Tukey HSD if significant
                    if p < self.alpha:
                        # Create array for Tukey's test
                        all_data = np.concatenate(group_data)
                        group_labels = np.concatenate([[g] * len(gd) for g, gd in zip(groups, group_data)])

                        # Perform Tukey's HSD test
                        tukey = pairwise_tukeyhsd(all_data, group_labels, alpha=self.alpha)

                        # Add post-hoc results
                        post_hoc_results = {
                            'post_hoc': 'Tukey HSD',
                            'post_hoc_details': str(tukey)
                        }
                    else:
                        post_hoc_results = {
                            'post_hoc': 'N/A',
                            'post_hoc_details': 'N/A'
                        }
                else:
                    # Kruskal-Wallis H test (non-parametric ANOVA)
                    stat, p = stats.kruskal(*group_data)
                    test_name = "Kruskal-Wallis H"

                    # Post-hoc Dunn's test if significant
                    if p < self.alpha:
                        post_hoc_results = {
                            'post_hoc': 'Dunn\'s test',
                            'post_hoc_details': 'See detailed results for pairwise comparisons'
                        }
                    else:
                        post_hoc_results = {
                            'post_hoc': 'N/A',
                            'post_hoc_details': 'N/A'
                        }

            # Format result
            result_dict = {
                'measure': measure,
                'test': test_name,
                'statistic': stat,
                'p_value': p,
                'significant': p < self.alpha,
            }

            # Add group-specific data
            if n_groups == 2:
                result_dict.update({
                    'n_group1': len(group1_data),
                    'n_group2': len(group2_data),
                    'mean_group1': group1_data.mean(),
                    'mean_group2': group2_data.mean(),
                    'std_group1': group1_data.std(),
                    'std_group2': group2_data.std(),
                    'effect_size': cohen_d,
                    'effect_size_type': 'Cohen\'s d'
                })
            else:
                # Add post-hoc results
                result_dict.update(post_hoc_results)

                # Add each group's stats
                for i, group in enumerate(groups):
                    result_dict[f'n_{group}'] = group_counts[i]
                    result_dict[f'mean_{group}'] = group_means[i]
                    result_dict[f'std_{group}'] = group_stds[i]

            # Append to results
            results.append(result_dict)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Save results if output directory specified
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"group_comparison_results_{timestamp}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"Group comparison results saved to: {output_path}")

        return results_df

    def correlate_volumes_with_clinical(self, volumes_df, clinical_df, subject_id_col='subject_id',
                                        volume_cols=None, clinical_cols=None, method='pearson'):
        """
        Correlate brain volumes with clinical measures.

        Args:
            volumes_df (pandas.DataFrame): DataFrame with brain volume data
            clinical_df (pandas.DataFrame): DataFrame with clinical data
            subject_id_col (str): Column name for subject ID
            volume_cols (list): List of column names with volume data
            clinical_cols (list): List of column names with clinical data
            method (str): Correlation method ('pearson' or 'spearman')

        Returns:
            pandas.DataFrame: DataFrame with correlation results
        """
        # Merge dataframes on subject ID
        merged_df = volumes_df.merge(clinical_df, on=subject_id_col, how='inner')

        # Identify volume columns if not specified
        if volume_cols is None:
            volume_cols = [col for col in volumes_df.columns
                           if ('volume' in col.lower() or 'class' in col.lower())
                           and col != subject_id_col]

        # Identify clinical columns if not specified
        if clinical_cols is None:
            # Common clinical measure names
            clinical_candidates = ['MMSE', 'ADAS', 'CDR', 'GDSCALE', 'FAQ', 'MOCA']
            clinical_cols = [col for col in clinical_df.columns
                             if any(candidate in col for candidate in clinical_candidates)
                             and col != subject_id_col]

        # Check if we have valid columns
        if not volume_cols or not clinical_cols:
            print("Error: No valid volume or clinical columns identified")
            return None

        # Initialize results
        correlation_results = []

        # Calculate correlations
        for vol_col in volume_cols:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(merged_df[vol_col]):
                continue

            for clin_col in clinical_cols:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(merged_df[clin_col]):
                    continue

                # Drop rows with NaN values
                valid_data = merged_df[[vol_col, clin_col]].dropna()

                # Check if we have enough data
                if len(valid_data) < 3:
                    correlation_results.append({
                        'volume_measure': vol_col,
                        'clinical_measure': clin_col,
                        'correlation': np.nan,
                        'p_value': np.nan,
                        'n': len(valid_data),
                        'significant': False
                    })
                    continue

                # Calculate correlation
                if method.lower() == 'pearson':
                    corr, p_value = pearsonr(valid_data[vol_col], valid_data[clin_col])
                elif method.lower() == 'spearman':
                    corr, p_value = spearmanr(valid_data[vol_col], valid_data[clin_col])
                else:
                    raise ValueError(f"Correlation method '{method}' not supported")

                # Add to results
                correlation_results.append({
                    'volume_measure': vol_col,
                    'clinical_measure': clin_col,
                    'correlation': corr,
                    'p_value': p_value,
                    'n': len(valid_data),
                    'significant': p_value < self.alpha
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(correlation_results)

        # Calculate FDR-corrected p-values
        if len(results_df) > 0:
            from statsmodels.stats.multitest import fdrcorrection
            _, corrected_pvals = fdrcorrection(results_df['p_value'].values)
            results_df['p_value_fdr'] = corrected_pvals
            results_df['significant_fdr'] = results_df['p_value_fdr'] < self.alpha

        # Save results if output directory specified
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"volume_clinical_correlations_{timestamp}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"Correlation results saved to: {output_path}")

        return results_df

    def classify_diagnostic_groups(self, features_df, diagnosis_col, feature_cols=None,
                                   subject_id_col='subject_id', classifier=None):
        """
        Perform diagnostic classification based on brain volumes or other features.

        Args:
            features_df (pandas.DataFrame): DataFrame with feature data
            diagnosis_col (str): Column name with diagnosis labels
            feature_cols (list): List of column names with features
            subject_id_col (str): Column name for subject ID
            classifier (sklearn classifier): Pre-initialized classifier (default: LogisticRegression)

        Returns:
            dict: Dictionary with classification results
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        from sklearn.pipeline import Pipeline

        # Identify feature columns if not specified
        if feature_cols is None:
            # Exclude ID and diagnosis columns
            feature_cols = [col for col in features_df.columns
                            if col != subject_id_col and col != diagnosis_col
                            and pd.api.types.is_numeric_dtype(features_df[col])]

        # Check if we have valid columns
        if not feature_cols:
            print("Error: No valid feature columns identified")
            return None

        # Prepare data
        X = features_df[feature_cols].values
        y = features_df[diagnosis_col].values

        # Set up classifier if not provided
        if classifier is None:
            classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

        # Set up pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy')

        # Train model on full dataset
        pipeline.fit(X, y)

        # Get predictions
        y_pred = pipeline.predict(X)
        y_prob = None

        # Get probability predictions if the classifier supports it
        if hasattr(pipeline, 'predict_proba'):
            y_prob = pipeline.predict_proba(X)

        # Compute metrics
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)

        # Compute ROC curve and AUC for each class
        roc_results = {}
        if y_prob is not None:
            classes = np.unique(y)
            for i, cls in enumerate(classes):
                # Binary case: class vs rest
                y_binary = (y == cls).astype(int)
                y_score = y_prob[:, i]

                fpr, tpr, _ = roc_curve(y_binary, y_score)
                roc_auc = auc(fpr, tpr)

                roc_results[cls] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                }

        # Prepare results
        results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': None,
            'roc_results': roc_results
        }

        # Get feature importance if available
        if hasattr(classifier, 'coef_'):
            results['feature_importance'] = {
                'names': feature_cols,
                'values': pipeline.named_steps['classifier'].coef_.tolist()
            }
        elif hasattr(classifier, 'feature_importances_'):
            results['feature_importance'] = {
                'names': feature_cols,
                'values': pipeline.named_steps['classifier'].feature_importances_.tolist()
            }

        # Save results if output directory specified
        if self.output_dir:
            # Save detailed results as JSON
            import json

            # Convert numpy arrays to lists for JSON serialization
            for key in results:
                if isinstance(results[key], np.ndarray):
                    results[key] = results[key].tolist()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"classification_results_{timestamp}.json"

            with open(output_path, 'w') as f:
                json.dump(results, f)

            print(f"Classification results saved to: {output_path}")

        return results

    def longitudinal_analysis(self, longitudinal_df, subject_id_col='subject_id', time_col='visit',
                              measure_cols=None, group_col=None):