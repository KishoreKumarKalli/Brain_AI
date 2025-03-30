import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from pylatex import Document, Section, Subsection, Figure, NoEscape, TableOfContents
from pylatex.basic import NewPage
from pylatex.section import Chapter
from pylatex.utils import bold
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
import yaml
import jinja2
import markdown
import json

# Import local modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.visualization import BrainMRIVisualizer
from analysis.statistics import BrainStatisticalAnalysis


class BrainAIReporting:
    """
    Class for generating comprehensive reports from brain MRI analysis results.
    Includes individual subject reports and group-level statistical reports.
    """

    def __init__(self, config_path='config.yml'):
        """Initialize the reporting class with configuration."""
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            self.config = {
                'data': {
                    'processed_dir': 'data/processed',
                    'clinical_dir': 'data/clinical'
                },
                'analysis': {
                    'significance_level': 0.05,
                    'volume_analysis': True,
                    'correlation_analysis': True,
                    'group_comparison': True,
                },
                'reporting': {
                    'output_dir': './reports',
                    'report_format': 'pdf',
                    'institution_name': 'Brain AI Research',
                    'logo_path': None,
                    'include_disclaimer': True
                }
            }

        # Define output directory
        self.output_dir = Path(self.config.get('reporting', {}).get('output_dir', './reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualization tool
        self.visualizer = BrainMRIVisualizer()

        # Date and time for reports
        self.report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Institution name for reports
        self.institution_name = self.config.get('reporting', {}).get('institution_name', 'Brain AI Research')

        # Disclaimer text
        self.disclaimer = ("This report is generated automatically by the Brain AI system and should be "
                           "reviewed by a qualified healthcare professional. The findings presented are based "
                           "on automated image analysis and are not intended to replace clinical judgment.")

        # Dictionary to store report templates
        self._templates = {
            'subject': None,
            'cohort': None,
            'longitudinal': None
        }

    def generate_subject_report(self, subject_id, original_image, segmentation_result,
                                clinical_data=None, anomaly_map=None, diagnosis=None,
                                output_format='pdf'):
        """
        Generate a comprehensive report for an individual subject.

        Args:
            subject_id (str): Subject identifier
            original_image (numpy.ndarray or str): Original MRI image array or path to NIfTI file
            segmentation_result (numpy.ndarray or str): Segmentation result array or path to NIfTI file
            clinical_data (dict or pandas.DataFrame): Subject's clinical data
            anomaly_map (numpy.ndarray): Anomaly detection map (if available)
            diagnosis (str): Clinical diagnosis (if available)
            output_format (str): 'pdf', 'html', or 'png'

        Returns:
            str: Path to the generated report file
        """
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{subject_id}_report_{timestamp}"

        # Load images if paths are provided
        if isinstance(original_image, str):
            original_image = nib.load(original_image).get_fdata()

        if isinstance(segmentation_result, str):
            segmentation_result = nib.load(segmentation_result).get_fdata()

        # Process clinical data if it's a DataFrame (extract row for this subject)
        if isinstance(clinical_data, pd.DataFrame):
            if 'subject_id' in clinical_data.columns:
                clinical_data = clinical_data[clinical_data['subject_id'] == subject_id].iloc[0].to_dict()
            elif 'PTID' in clinical_data.columns:
                clinical_data = clinical_data[clinical_data['PTID'] == subject_id].iloc[0].to_dict()
            else:
                print("Warning: Could not find subject in clinical data DataFrame")
                clinical_data = {}

        # Calculate volumetric measurements
        volumes = self._calculate_volumes(segmentation_result)

        # Create PDF report
        if output_format.lower() == 'pdf':
            return self._generate_pdf_subject_report(
                subject_id, original_image, segmentation_result,
                volumes, clinical_data, anomaly_map, diagnosis, report_filename
            )

        # Create HTML report
        elif output_format.lower() == 'html':
            return self._generate_html_subject_report(
                subject_id, original_image, segmentation_result,
                volumes, clinical_data, anomaly_map, diagnosis, report_filename
            )

        # Create PNG report
        else:  # default to PNG
            return self._generate_png_subject_report(
                subject_id, original_image, segmentation_result,
                volumes, clinical_data, anomaly_map, diagnosis, report_filename
            )

    def generate_cohort_report(self, results_df, clinical_df=None, group_column='DX_GROUP',
                               output_format='pdf'):
        """
        Generate a comprehensive report comparing metrics across different groups.

        Args:
            results_df (pandas.DataFrame): DataFrame with volumetric results
            clinical_df (pandas.DataFrame, optional): DataFrame with clinical data
            group_column (str): Column name for grouping (e.g., 'DX_GROUP')
            output_format (str): 'pdf', 'html', or 'png'

        Returns:
            str: Path to the generated report file
        """
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"cohort_analysis_report_{timestamp}"

        # Merge with clinical data if provided
        if clinical_df is not None:
            # Find common ID column
            common_id_cols = set(results_df.columns).intersection(set(clinical_df.columns))
            id_candidates = [col for col in common_id_cols if any(id_term in col.lower()
                                                                  for id_term in ['id', 'subject', 'patient', 'ptid'])]

            if id_candidates:
                merged_df = pd.merge(results_df, clinical_df, on=id_candidates[0], how='inner')
            else:
                print("Warning: No common ID column found between results and clinical data")
                merged_df = results_df
        else:
            merged_df = results_df

        # Make sure we have the group column
        if group_column not in merged_df.columns:
            print(f"Warning: Group column '{group_column}' not found in data")
            # Try to find a suitable replacement
            group_candidates = [col for col in merged_df.columns
                                if any(term in col.lower() for term in
                                       ['group', 'diagnosis', 'dx', 'class', 'category'])]
            if group_candidates:
                group_column = group_candidates[0]
                print(f"Using '{group_column}' as group column instead")
            else:
                print("No suitable group column found. Cannot generate group comparison report.")
                return None

        # Perform statistical analysis
        stats_analyzer = BrainStatisticalAnalysis()

        # Find volume columns
        volume_cols = [col for col in merged_df.columns
                       if any(vol_term in col.lower() for vol_term in ['volume', 'vol'])
                       and col not in [group_column]]

        # Find clinical score columns
        clinical_cols = [col for col in merged_df.columns
                         if any(term in col.upper() for term in
                                ['MMSE', 'CDR', 'GDSCALE', 'ADAS', 'FAQ', 'MOCA'])
                         and col not in [group_column]]

        # Create report based on format
        if output_format.lower() == 'pdf':
            return self._generate_pdf_cohort_report(
                merged_df, volume_cols, clinical_cols, group_column, stats_analyzer, report_filename
            )
        elif output_format.lower() == 'html':
            return self._generate_html_cohort_report(
                merged_df, volume_cols, clinical_cols, group_column, stats_analyzer, report_filename
            )
        else:  # default to PNG
            return self._generate_png_cohort_report(
                merged_df, volume_cols, clinical_cols, group_column, stats_analyzer, report_filename
            )

    def generate_longitudinal_report(self, subject_id, timepoints_data, clinical_data=None,
                                     output_format='pdf'):
        """
        Generate a longitudinal analysis report for a subject across multiple timepoints.

        Args:
            subject_id (str): Subject identifier
            timepoints_data (list): List of dictionaries containing timepoint data
                Each dict should have:
                - 'timepoint': timepoint identifier (e.g., 'baseline', 'month_12')
                - 'image': Original MRI image array or path
                - 'segmentation': Segmentation result array or path
                - 'date': Scan date (optional)
            clinical_data (pandas.DataFrame): DataFrame with clinical data across timepoints
            output_format (str): 'pdf', 'html', or 'png'

        Returns:
            str: Path to the generated report file
        """
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{subject_id}_longitudinal_report_{timestamp}"

        # Process timepoints data
        processed_timepoints = []
        for tp in timepoints_data:
            tp_data = tp.copy()

            # Load images if paths are provided
            if isinstance(tp['image'], str):
                tp_data['image'] = nib.load(tp['image']).get_fdata()

            if isinstance(tp['segmentation'], str):
                tp_data['segmentation'] = nib.load(tp['segmentation']).get_fdata()

            # Calculate volumes
            tp_data['volumes'] = self._calculate_volumes(tp_data['segmentation'])
            processed_timepoints.append(tp_data)

        # Extract clinical data for this subject if provided as DataFrame
        subject_clinical = None
        if isinstance(clinical_data, pd.DataFrame):
            if 'subject_id' in clinical_data.columns and 'timepoint' in clinical_data.columns:
                subject_clinical = clinical_data[clinical_data['subject_id'] == subject_id]
            elif 'PTID' in clinical_data.columns and 'VISCODE' in clinical_data.columns:
                subject_clinical = clinical_data[clinical_data['PTID'] == subject_id]

        # Create report based on format
        if output_format.lower() == 'pdf':
            return self._generate_pdf_longitudinal_report(
                subject_id, processed_timepoints, subject_clinical, report_filename
            )
        elif output_format.lower() == 'html':
            return self._generate_html_longitudinal_report(
                subject_id, processed_timepoints, subject_clinical, report_filename
            )
        else:  # default to PNG
            return self._generate_png_longitudinal_report(
                subject_id, processed_timepoints, subject_clinical, report_filename
            )