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

    def longitudinal_analysis(self, longitudinal_df, volume_cols, time_col='EXAMDATE',
                              subject_id_col='PTID', group_col='DX_GROUP',
                              output_dir=None):
        """
        Analyze longitudinal changes in brain volumes over time.

        Args:
            longitudinal_df (pandas.DataFrame): DataFrame containing longitudinal data
            volume_cols (list): List of columns containing volume measurements
            time_col (str): Column name containing examination date/time
            subject_id_col (str): Column name containing subject identifiers
            group_col (str): Column name containing diagnostic group labels
            output_dir (str, optional): Directory to save output files

        Returns:
            dict: Dictionary containing analysis results
        """
        # Ensure time column is properly formatted as datetime
        if pd.api.types.is_string_dtype(longitudinal_df[time_col]):
            try:
                longitudinal_df[time_col] = pd.to_datetime(longitudinal_df[time_col])
            except:
                print(f"Warning: Could not convert {time_col} to datetime. Treating as categorical.")

        # Sort by subject ID and date
        longitudinal_df = longitudinal_df.sort_values([subject_id_col, time_col])

        # Calculate time since baseline for each subject
        longitudinal_df['time_since_baseline'] = longitudinal_df.groupby(subject_id_col)[time_col].transform(
            lambda x: (x - x.min()).dt.days / 365.25 if pd.api.types.is_datetime64_dtype(x) else 0
        )

        # Identify subjects with multiple time points
        visit_counts = longitudinal_df[subject_id_col].value_counts()
        subjects_with_followup = visit_counts[visit_counts > 1].index.tolist()

        if len(subjects_with_followup) == 0:
            print("No subjects with longitudinal data found.")
            return None

        # Filter to only include subjects with follow-up data
        long_data = longitudinal_df[longitudinal_df[subject_id_col].isin(subjects_with_followup)]

        # Dictionary to store results
        results = {
            'rate_of_change': {},
            'group_differences': {},
            'mixed_effects': {},
            'visualization': {}
        }

        # Calculate annual rate of change for each subject and region
        rate_of_change_df = []

        for subject in subjects_with_followup:
            subject_data = long_data[long_data[subject_id_col] == subject]

            # Skip if less than 2 time points
            if len(subject_data) < 2:
                continue

            # Get baseline values
            baseline = subject_data.loc[subject_data['time_since_baseline'].idxmin()]

            # Get last follow-up values
            last_followup = subject_data.loc[subject_data['time_since_baseline'].idxmax()]

            # Calculate time difference in years
            time_diff = last_followup['time_since_baseline'] - baseline['time_since_baseline']

            if time_diff <= 0:
                continue

            # Get diagnostic group (use last visit's diagnosis as current status)
            if group_col in subject_data.columns:
                diagnosis = last_followup[group_col]
            else:
                diagnosis = 'Unknown'

            # Calculate rate of change for each volume
            for col in volume_cols:
                if col not in baseline or col not in last_followup:
                    continue

                # Calculate absolute and percentage change
                absolute_change = last_followup[col] - baseline[col]
                percent_change = (absolute_change / baseline[col]) * 100 if baseline[col] != 0 else np.nan

                # Calculate annualized rate of change
                annual_absolute_change = absolute_change / time_diff
                annual_percent_change = percent_change / time_diff

                # Add to results dataframe
                rate_of_change_df.append({
                    subject_id_col: subject,
                    'brain_region': col.replace('_volume', '').replace('_mm3', ''),
                    'baseline_volume': baseline[col],
                    'followup_volume': last_followup[col],
                    'absolute_change': absolute_change,
                    'percent_change': percent_change,
                    'follow_up_years': time_diff,
                    'annual_absolute_change': annual_absolute_change,
                    'annual_percent_change': annual_percent_change,
                    'diagnosis': diagnosis
                })

        # Convert to DataFrame
        rate_of_change_df = pd.DataFrame(rate_of_change_df)
        results['rate_of_change']['data'] = rate_of_change_df

        if len(rate_of_change_df) == 0:
            print("No valid longitudinal measurements found.")
            return results

        # Create summary statistics of annualized rates of change
        summary_stats = rate_of_change_df.groupby(['brain_region', 'diagnosis']).agg({
            'annual_absolute_change': ['mean', 'std', 'min', 'max', 'count'],
            'annual_percent_change': ['mean', 'std', 'min', 'max']
        })

        results['rate_of_change']['summary'] = summary_stats

        # Compare atrophy rates between diagnostic groups
        if group_col in longitudinal_df.columns and len(longitudinal_df[group_col].unique()) > 1:
            group_diff_results = {}

            for region in rate_of_change_df['brain_region'].unique():
                region_data = rate_of_change_df[rate_of_change_df['brain_region'] == region]

                if len(region_data) <= 1:
                    continue

                try:
                    # One-way ANOVA for group differences in atrophy rate
                    formula = 'annual_percent_change ~ diagnosis'
                    model = ols(formula, data=region_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # Tukey's HSD post-hoc test for pairwise comparisons
                    tukey = pairwise_tukeyhsd(
                        endog=region_data['annual_percent_change'],
                        groups=region_data['diagnosis'],
                        alpha=0.05
                    )

                    group_diff_results[region] = {
                        'anova': anova_table,
                        'tukey_hsd': tukey
                    }
                except Exception as e:
                    print(f"Error in group comparison for {region}: {e}")

            results['group_differences'] = group_diff_results

        # Mixed-effects models for longitudinal analysis
        if len(subjects_with_followup) >= 10:  # Need sufficient subjects for mixed-effects
            try:
                import statsmodels.formula.api as smf
                from statsmodels.regression.mixed_linear_model import MixedLM

                mixed_effects_results = {}

                # Prepare data for mixed models (all time points)
                mixed_model_data = longitudinal_df[longitudinal_df[subject_id_col].isin(subjects_with_followup)]

                for vol_col in volume_cols:
                    # Skip if column doesn't exist
                    if vol_col not in mixed_model_data.columns:
                        continue

                    try:
                        # Basic mixed model with random intercept
                        # Volume ~ time + (1|subject)
                        formula = f"{vol_col} ~ time_since_baseline"

                        # Add diagnostic group as fixed effect if available
                        if group_col in mixed_model_data.columns:
                            formula += f" + {group_col} + time_since_baseline:{group_col}"

                        # Fit mixed model
                        md = smf.mixedlm(
                            formula,
                            mixed_model_data,
                            groups=mixed_model_data[subject_id_col]
                        )

                        mdf = md.fit()

                        # Store results
                        region_name = vol_col.replace('_volume', '').replace('_mm3', '')
                        mixed_effects_results[region_name] = {
                            'model_summary': mdf.summary(),
                            'params': mdf.params.to_dict(),
                            'pvalues': mdf.pvalues.to_dict()
                        }
                    except Exception as e:
                        print(f"Error in mixed model for {vol_col}: {e}")

                results['mixed_effects'] = mixed_effects_results
            except ImportError:
                print("Warning: statsmodels not available for mixed-effects modeling")

        # Create visualizations for longitudinal data
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Plot atrophy rates by diagnostic group
            if group_col in rate_of_change_df.columns and len(rate_of_change_df[group_col].unique()) > 1:
                try:
                    for region in rate_of_change_df['brain_region'].unique():
                        region_data = rate_of_change_df[rate_of_change_df['brain_region'] == region]

                        # Skip regions with insufficient data
                        if len(region_data) <= 3:
                            continue

                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Create boxplot of annual percent change by diagnosis
                        sns.boxplot(x='diagnosis', y='annual_percent_change', data=region_data, ax=ax)
                        sns.stripplot(x='diagnosis', y='annual_percent_change', data=region_data,
                                      color='black', alpha=0.5, ax=ax)

                        ax.set_title(f'Annual % Change in {region} Volume by Diagnostic Group')
                        ax.set_ylabel('Annual % Change')
                        ax.set_xlabel('Diagnostic Group')

                        # Add horizontal line at zero (no change)
                        ax.axhline(y=0, color='r', linestyle='--')

                        # Calculate mean values for text annotations
                        means = region_data.groupby('diagnosis')['annual_percent_change'].mean()

                        # Add mean values as text on plot
                        for i, group in enumerate(means.index):
                            ax.text(i, means[group], f'Mean: {means[group]:.2f}%',
                                    ha='center', va='bottom', fontweight='bold')

                        plt.tight_layout()

                        # Save figure
                        fig_path = os.path.join(output_dir, f'longitudinal_{region}_change.png')
                        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close()

                        # Store path in results
                        if 'figures' not in results['visualization']:
                            results['visualization']['figures'] = []

                        results['visualization']['figures'].append(fig_path)
                except Exception as e:
                    print(f"Error creating atrophy rate plots: {e}")

            # Create spaghetti plots showing individual trajectories
            try:
                for vol_col in volume_cols:
                    if vol_col not in longitudinal_df.columns:
                        continue

                    fig, ax = plt.subplots(figsize=(12, 8))

                    # Plot individual subject trajectories
                    for subject in subjects_with_followup:
                        subject_data = longitudinal_df[longitudinal_df[subject_id_col] == subject]

                        if len(subject_data) < 2:
                            continue

                        # Determine line color based on diagnostic group if available
                        if group_col in subject_data.columns:
                            diagnosis = subject_data[group_col].iloc[-1]  # Use last diagnosis

                            # Create color map for diagnostic groups
                            unique_dx = longitudinal_df[group_col].unique()
                            cmap = plt.cm.get_cmap('viridis', len(unique_dx))
                            color_map = {dx: cmap(i) for i, dx in enumerate(unique_dx)}

                            line_color = color_map.get(diagnosis, 'gray')
                            line_alpha = 0.6
                        else:
                            line_color = 'steelblue'
                            line_alpha = 0.3

                        # Plot trajectory
                        ax.plot(subject_data['time_since_baseline'], subject_data[vol_col],
                                'o-', alpha=line_alpha, color=line_color)

                    # Add group mean trajectories
                    if group_col in longitudinal_df.columns:
                        for group in longitudinal_df[group_col].unique():
                            group_data = longitudinal_df[longitudinal_df[group_col] == group]

                            # Group by time (rounded to nearest 0.5 year)
                            group_data['time_rounded'] = np.round(group_data['time_since_baseline'] * 2) / 2
                            mean_trajectory = group_data.groupby('time_rounded')[vol_col].mean().reset_index()

                            if len(mean_trajectory) > 1:
                                ax.plot(mean_trajectory['time_rounded'], mean_trajectory[vol_col],
                                        'o-', linewidth=3, markersize=8,
                                        color=color_map.get(group, 'black'),
                                        label=f'{group} Mean')

                    # Set labels and title
                    region_name = vol_col.replace('_volume', '').replace('_mm3', '')
                    ax.set_title(f'{region_name} Volume Over Time')
                    ax.set_xlabel('Time Since Baseline (years)')
                    ax.set_ylabel('Volume (mm³)')

                    # Add legend if groups exist
                    if group_col in longitudinal_df.columns:
                        ax.legend(title='Diagnostic Group')

                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save figure
                    fig_path = os.path.join(output_dir, f'longitudinal_{region_name}_trajectories.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    # Store path in results
                    if 'figures' not in results['visualization']:
                        results['visualization']['figures'] = []

                    results['visualization']['figures'].append(fig_path)

            except Exception as e:
                print(f"Error creating trajectory plots: {e}")

        # Save results to CSV
        if output_dir and len(rate_of_change_df) > 0:
            rate_of_change_df.to_csv(os.path.join(output_dir, 'longitudinal_rate_of_change.csv'), index=False)

        return results

    def classification_analysis(self, volumetric_data, clinical_data=None, target_column='DX_GROUP',
                                features=None, models=None, cv_folds=5, output_dir=None):
        """
        Perform classification analysis to predict diagnostic groups based on brain volumes.

        Args:
            volumetric_data (pd.DataFrame): DataFrame with volumetric measurements
            clinical_data (pd.DataFrame, optional): DataFrame with clinical measurements
            target_column (str): Column name containing the target classes
            features (list, optional): List of features to use for classification
            models (dict, optional): Dictionary of model names and their instances
            cv_folds (int): Number of cross-validation folds
            output_dir (str, optional): Directory to save the results

        Returns:
            dict: Dictionary containing classification results
        """
        print("Performing classification analysis...")

        # Merge data if clinical data is provided
        if clinical_data is not None:
            # Find common ID column
            common_cols = set(volumetric_data.columns) & set(clinical_data.columns)
            id_col = next((col for col in common_cols
                           if any(id_term in col.lower() for id_term in ['id', 'subject'])), None)

            if id_col:
                data = pd.merge(volumetric_data, clinical_data, on=id_col, how='inner')
            else:
                print("Warning: No common ID column found. Using only volumetric data.")
                data = volumetric_data.copy()
        else:
            data = volumetric_data.copy()

        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data")

        # Drop rows with missing target values
        data = data.dropna(subset=[target_column])

        # Convert target to string (categorical)
        data[target_column] = data[target_column].astype(str)

        # Get unique classes
        classes = data[target_column].unique()
        if len(classes) < 2:
            raise ValueError(f"Need at least 2 classes for classification, found {len(classes)}")

        # Select features if not specified
        if features is None:
            # Use volumetric features by default
            features = [col for col in data.columns if any(term in col.lower()
                                                           for term in ['volume', 'class_', 'region_'])
                        and col != 'total_brain_volume']

        # Check if we have enough features
        if len(features) == 0:
            raise ValueError("No suitable features found for classification")

        # Remove any features with missing values
        data = data.dropna(subset=features)

        # Prepare X and y
        X = data[features].values
        y = data[target_column].values

        # Check if we have enough samples
        if len(X) < cv_folds * len(classes):
            print(f"Warning: Very small sample size ({len(X)} samples) for {len(classes)} classes "
                  f"and {cv_folds} CV folds. Results may not be reliable.")

        # Set up models if not provided
        if models is None:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
                'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
                'SVM': SVC(probability=True, class_weight='balanced')
            }

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        results = {}

        # Create figure for ROC curves
        plt.figure(figsize=(12, 8))

        # Evaluate each model
        for name, model in models.items():
            print(f"Evaluating {name}...")

            # Cross-validation accuracy
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            mean_accuracy = cv_scores.mean()
            std_accuracy = cv_scores.std()

            print(f"{name} CV Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")

            # For binary classification, compute ROC AUC
            if len(classes) == 2:
                # Get probabilities from cross-validation
                y_probs = np.zeros_like(y, dtype=float)

                for train_idx, test_idx in cv.split(X, y):
                    model.fit(X[train_idx], y[train_idx])
                    y_probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y == classes[1], y_probs)
                auc_score = roc_auc_score(y == classes[1], y_probs)

                # Plot ROC curve
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')

                results[name] = {
                    'accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'auc': auc_score,
                    'fpr': fpr,
                    'tpr': tpr
                }
            else:
                # For multi-class, use one-vs-rest ROC AUC
                y_probs = np.zeros((len(y), len(classes)))

                for train_idx, test_idx in cv.split(X, y):
                    model.fit(X[train_idx], y[train_idx])
                    y_probs[test_idx] = model.predict_proba(X[test_idx])

                # Calculate macro-average ROC AUC
                auc_scores = []
                for i, class_name in enumerate(classes):
                    auc_score = roc_auc_score((y == class_name).astype(int), y_probs[:, i])
                    auc_scores.append(auc_score)

                macro_auc = np.mean(auc_scores)

                results[name] = {
                    'accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'macro_auc': macro_auc,
                    'class_auc_scores': dict(zip(classes, auc_scores))
                }

        # Complete ROC curve plot
        if len(classes) == 2:
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')

            # Save ROC curve if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')

        plt.close()

        # Feature importance analysis
        feature_importances = self._analyze_feature_importance(X, y, features, models)
        results['feature_importances'] = feature_importances

        # Create feature importance plot
        self._plot_feature_importance(feature_importances, output_dir)

        # Final model training on full dataset
        final_model = models['Random Forest'] if 'Random Forest' in models else list(models.values())[0]
        final_model.fit(X, y)

        # Generate confusion matrix
        y_pred = final_model.predict(X)
        conf_matrix = confusion_matrix(y, y_pred)

        # Normalize confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')

        # Save confusion matrix if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')

        plt.close()

        # Generate classification report
        class_report = classification_report(y, y_pred, target_names=classes, output_dict=True)
        results['classification_report'] = class_report

        # Save results to CSV if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save model performance summary
            performance_df = pd.DataFrame({
                'Model': list(models.keys()),
                'Accuracy': [results[name]['accuracy'] for name in models.keys()],
                'Std_Accuracy': [results[name]['std_accuracy'] for name in models.keys()]
            })

            if len(classes) == 2:
                performance_df['AUC'] = [results[name]['auc'] for name in models.keys()]
            else:
                performance_df['Macro_AUC'] = [results[name]['macro_auc'] for name in models.keys()]

            performance_df.to_csv(os.path.join(output_dir, 'model_performance.csv'), index=False)

            # Save classification report
            report_df = pd.DataFrame(class_report).transpose()
            report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))

            # Save feature importances
            feature_imp_df = pd.DataFrame(feature_importances)
            feature_imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

        return results

    def _analyze_feature_importance(self, X, y, feature_names, models):
        """
        Analyze and extract feature importance from different models.

        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            feature_names (list): List of feature names
            models (dict): Dictionary of model names and their instances

        Returns:
            dict: Dictionary containing feature importance for each model
        """
        feature_importances = {}

        for name, model in models.items():
            # Fit model
            model.fit(X, y)

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):  # e.g., Random Forest
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):  # e.g., Linear models
                importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                # Skip if model doesn't provide feature importance
                continue

            # Create sorted list of feature importances
            features_importance_dict = dict(zip(feature_names, importances))
            sorted_features = sorted(features_importance_dict.items(), key=lambda x: x[1], reverse=True)

            feature_importances[name] = {
                'features': [item[0] for item in sorted_features],
                'importance': [item[1] for item in sorted_features]
            }

        return feature_importances

    def _plot_feature_importance(self, feature_importances, output_dir=None):
        """
        Plot feature importance for each model.

        Args:
            feature_importances (dict): Dictionary containing feature importance for each model
            output_dir (str, optional): Directory to save the plot

        Returns:
            matplotlib.figure.Figure: Feature importance plot
        """
        # Check if feature importances exist
        if not feature_importances:
            return None

        # Determine number of models
        n_models = len(feature_importances)

        # Create figure
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 6 * n_models))

        # Convert to array if only one model
        if n_models == 1:
            axes = [axes]

        # Plot feature importance for each model
        for i, (name, importance) in enumerate(feature_importances.items()):
            ax = axes[i]

            # Get features and importance scores
            features = importance['features']
            scores = importance['importance']

            # Limit to top 15 features for readability
            if len(features) > 15:
                features = features[:15]
                scores = scores[:15]

            # Plot horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('_volume_mm3', '').replace('_', ' ').title() for f in features])
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top Features - {name}')

        plt.tight_layout()

        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')

        return fig

    def summarize_group_differences(self, data_df, volume_columns=None, group_column='DX_GROUP',
                                    output_path=None, alpha=0.05):
        """
        Perform statistical comparison of brain volumes between diagnostic groups.

        Args:
            data_df (pandas.DataFrame): DataFrame containing volume and group data
            volume_columns (list, optional): List of volume column names to analyze
            group_column (str): Column name containing group labels
            output_path (str, optional): Path to save the results
            alpha (float): Significance level for statistical tests

        Returns:
            pandas.DataFrame: Summary statistics of group differences
        """
        # Validate inputs
        if group_column not in data_df.columns:
            raise ValueError(f"Group column '{group_column}' not found in DataFrame")

        # Identify volume columns if not provided
        if volume_columns is None:
            volume_columns = [col for col in data_df.columns
                              if 'volume' in col.lower() and col != 'total_volume']

        # Check if there are enough groups for comparison
        groups = data_df[group_column].unique()
        if len(groups) < 2:
            print("Warning: At least 2 groups needed for comparison. Returning descriptive statistics.")
            return data_df.groupby(group_column)[volume_columns].describe()

        # Create results dataframe
        results = pd.DataFrame(columns=[
            'Region', 'Test', 'p_value', 'Significant', 'Effect_Size',
            'Group_Means', 'Post_Hoc'
        ])

        # Analyze each volume column
        for vol_col in volume_columns:
            # Skip if column doesn't exist
            if vol_col not in data_df.columns:
                continue

            # Prepare data for this volume
            data = data_df.dropna(subset=[vol_col, group_column])

            # Skip if not enough data
            if len(data) < 5:
                continue

            # Format region name for output
            region_name = vol_col.replace('_volume', '').replace('_mm3', '').replace('class_', '').title()

            # Check for normality using Shapiro-Wilk test for each group
            # If any group has non-normal distribution or if any group has < 3 samples,
            # use non-parametric Kruskal-Wallis, otherwise use ANOVA
            normal_distribution = True
            small_groups = False

            for group in groups:
                group_data = data[data[group_column] == group][vol_col]
                if len(group_data) < 3:
                    small_groups = True
                elif len(group_data) < 50:  # Only test normality for reasonably sized samples
                    _, p = stats.shapiro(group_data)
                    if p < 0.05:  # Non-normal distribution
                        normal_distribution = False

            # Group means for reporting
            group_means = data.groupby(group_column)[vol_col].mean().to_dict()
            group_means_str = ", ".join([f"{g}: {v:.2f}" for g, v in group_means.items()])

            # Choose appropriate test based on data characteristics
            if len(groups) == 2:
                # Two groups: t-test or Mann-Whitney
                g1 = data[data[group_column] == groups[0]][vol_col]
                g2 = data[data[group_column] == groups[1]][vol_col]

                if normal_distribution and not small_groups:
                    # Check for equal variances
                    _, var_p = stats.levene(g1, g2)
                    equal_var = var_p >= 0.05

                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)

                    # Calculate Cohen's d effect size
                    pooled_std = np.sqrt(((len(g1) - 1) * g1.std() ** 2 +
                                          (len(g2) - 1) * g2.std() ** 2) /
                                         (len(g1) + len(g2) - 2))
                    effect_size = abs((g1.mean() - g2.mean()) / pooled_std) if pooled_std != 0 else 0
                    effect_type = "Cohen's d"

                    # Test name based on variance equality
                    test_name = "Two-sample t-test (equal var)" if equal_var else "Welch's t-test (unequal var)"
                    post_hoc = "N/A"
                else:
                    # Non-parametric Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(g1, g2)

                    # Calculate non-parametric effect size (r = Z/sqrt(N))
                    n1, n2 = len(g1), len(g2)
                    z_score = u_stat / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                    effect_size = abs(z_score / np.sqrt(n1 + n2))
                    effect_type = "r (non-parametric)"

                    test_name = "Mann-Whitney U test"
                    post_hoc = "N/A"
            else:
                # More than two groups: ANOVA or Kruskal-Wallis
                if normal_distribution and not small_groups:
                    # Perform one-way ANOVA
                    groups_list = [data[data[group_column] == g][vol_col] for g in groups]
                    f_stat, p_value = stats.f_oneway(*groups_list)

                    # Calculate eta-squared effect size
                    # First create ANOVA model with statsmodels for SS values
                    model = ols(f"{vol_col} ~ C({group_column})", data=data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # Calculate eta-squared
                    ss_between = anova_table.loc[f'C({group_column})', 'sum_sq']
                    ss_total = ss_between + anova_table.loc['Residual', 'sum_sq']
                    effect_size = ss_between / ss_total if ss_total != 0 else 0
                    effect_type = "Eta-squared"

                    test_name = "One-way ANOVA"

                    # Perform post-hoc Tukey test if ANOVA is significant
                    if p_value < alpha:
                        try:
                            posthoc = pairwise_tukeyhsd(data[vol_col], data[group_column], alpha=alpha)
                            posthoc_results = []

                            # Format post-hoc results
                            for i, (group1, group2, reject) in enumerate(zip(posthoc.data[0],
                                                                             posthoc.data[1],
                                                                             posthoc.reject)):
                                if reject:
                                    posthoc_results.append(f"{group1} vs {group2}: Significant")

                            post_hoc = "; ".join(posthoc_results) if posthoc_results else "No significant pairs"
                        except:
                            post_hoc = "Error in post-hoc test"
                    else:
                        post_hoc = "ANOVA not significant"
                else:
                    # Perform Kruskal-Wallis H-test
                    h_stat, p_value = stats.kruskal(*[data[data[group_column] == g][vol_col]
                                                      for g in groups])

                    # Calculate epsilon-squared effect size for Kruskal-Wallis
                    n = len(data)
                    k = len(groups)
                    effect_size = (h_stat - k + 1) / (n - k) if n != k else 0
                    effect_type = "Epsilon-squared"

                    test_name = "Kruskal-Wallis H-test"

                    # Perform post-hoc Dunn's test if significant
                    if p_value < alpha:
                        try:
                            from scikit_posthocs import posthoc_dunn

                            # Run Dunn's test
                            dunn_results = posthoc_dunn(data, val_col=vol_col, group_col=group_column,
                                                        p_adjust='bonferroni')

                            # Format post-hoc results
                            posthoc_results = []
                            for i, g1 in enumerate(groups):
                                for j, g2 in enumerate(groups):
                                    if i < j:  # Avoid duplicates
                                        p = dunn_results[g1][g2]
                                        if p < alpha:
                                            posthoc_results.append(f"{g1} vs {g2}: p={p:.3f}")

                            post_hoc = "; ".join(posthoc_results) if posthoc_results else "No significant pairs"
                        except:
                            post_hoc = "Error in post-hoc test"
                    else:
                        post_hoc = "Kruskal-Wallis not significant"

            # Add result to the dataframe
            results = results.append({
                'Region': region_name,
                'Test': test_name,
                'p_value': p_value,
                'Significant': p_value < alpha,
                'Effect_Size': f"{effect_size:.3f} ({effect_type})",
                'Group_Means': group_means_str,
                'Post_Hoc': post_hoc
            }, ignore_index=True)

        # Save results to CSV if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results.to_csv(output_path, index=False)
            print(f"Group difference results saved to {output_path}")

        return results

    def calculate_volume_percentiles(self, volumes_df, reference_group='CN',
                                     volume_columns=None, group_column='DX_GROUP',
                                     output_path=None):
        """
        Calculate percentiles for brain volumes using a reference group.

        Args:
            volumes_df (pandas.DataFrame): DataFrame with volume data
            reference_group (str): Label of the reference group (e.g., 'CN' for Control Normal)
            volume_columns (list, optional): List of volume column names
            group_column (str): Column name for group labels
            output_path (str, optional): Path to save the output

        Returns:
            pandas.DataFrame: Subject-level percentile data
        """
        # Verify reference group exists in data
        if group_column not in volumes_df.columns:
            raise ValueError(f"Group column '{group_column}' not found in DataFrame")

        if reference_group not in volumes_df[group_column].values:
            raise ValueError(f"Reference group '{reference_group}' not found in data")

        # Identify volume columns if not provided
        if volume_columns is None:
            volume_columns = [col for col in volumes_df.columns
                              if 'volume' in col.lower() and col != 'total_volume']

        # Get reference data
        reference_df = volumes_df[volumes_df[group_column] == reference_group]

        # Create a new dataframe for percentiles
        result_df = volumes_df.copy()

        # Calculate percentiles for each volume column
        for vol_col in volume_columns:
            if vol_col in volumes_df.columns:
                # Create new column for percentiles
                percentile_col = f"{vol_col}_percentile"

                # Calculate percentile for each subject based on reference distribution
                result_df[percentile_col] = result_df[vol_col].apply(
                    lambda x: stats.percentileofscore(reference_df[vol_col].dropna(), x)
                )

                # Create categorical assessment based on percentiles
                assessment_col = f"{vol_col}_assessment"

                # Define function to categorize the percentile
                def categorize_percentile(p):
                    if p < 5:
                        return "Very Low (< 5th)"
                    elif p < 25:
                        return "Low (5th-25th)"
                    elif p < 75:
                        return "Normal (25th-75th)"
                    elif p < 95:
                        return "High (75th-95th)"
                    else:
                        return "Very High (> 95th)"

                result_df[assessment_col] = result_df[percentile_col].apply(categorize_percentile)

        # Save results to CSV if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"Volume percentile results saved to {output_path}")

        return result_df

    def build_predictive_model(self, data_df, target_column='DX_GROUP', feature_columns=None,
                               model_type='random_forest', output_path=None, cv=5):
        """
        Build and evaluate machine learning models to predict diagnostic group
        from brain volume measurements.

        Args:
            data_df (pandas.DataFrame): DataFrame containing features and target
            target_column (str): Column name of the target variable
            feature_columns (list, optional): List of feature column names
            model_type (str): Type of model to build ('random_forest', 'svm', or 'logistic')
            output_path (str, optional): Path to save model evaluation results
            cv (int): Number of cross-validation folds

        Returns:
            tuple: Model, cross-validation scores, and feature importance
        """
        # Validate inputs
        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Identify feature columns if not provided
        if feature_columns is None:
            feature_columns = [col for col in data_df.columns
                               if ('volume' in col.lower() or 'ratio' in col.lower())
                               and 'percentile' not in col.lower()
                               and 'assessment' not in col.lower()]

        # Drop rows with NaN in target or features
        analysis_df = data_df.dropna(subset=[target_column] + feature_columns)

        # Check if we have enough data
        if len(analysis_df) < 10:
            print("Warning: Not enough data for model building (n < 10)")
            return None, None, None

        # Prepare X and y
        X = analysis_df[feature_columns].values
        y = analysis_df[target_column].values

        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select and configure model based on type
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        elif model_type == 'svm':
            model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        elif model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cross-validation
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_obj, scoring='accuracy')

        # Train final model on all data
        model.fit(X_scaled, y)

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_columns, model.feature_importances_))

        # Generate detailed evaluation if output path is provided
        if output_path:
            self.evaluate_and_save_model(model, X_scaled, y, feature_columns, output_path)

        return model, cv_scores, feature_importance

    def evaluate_and_save_model(self, model, X, y, feature_names, output_path):
        """
        Perform detailed model evaluation and save results.

        Args:
            model: Trained classifier model
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            output_path: Path to save evaluation results
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, classification_report,
                                     roc_curve, roc_auc_score)
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Make predictions
        y_pred = model.predict(X)

        # For ROC curve, need predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        else:
            y_prob = None

        # Create directory for output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Basic metrics
        accuracy = accuracy_score(y, y_pred)

        # For multi-class problems, use weighted averaging
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        # Save detailed classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(f"{output_path}_classification_report.csv")

        # Create confusion matrix visualization
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{output_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Feature importance visualization if available
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance')
            plt.bar(range(len(indices)), importance[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"{output_path}_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()

        # ROC curve for binary classification or one-vs-rest for multiclass
        if y_prob is not None:
            # Check if binary or multiclass
            unique_classes = np.unique(y)
            if len(unique_classes) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y, y_prob[:, 1])
                roc_auc = roc_auc_score(y, y_prob[:, 1])

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                plt.savefig(f"{output_path}_roc_curve.png", dpi=300, bbox_inches='tight')
                plt.close()