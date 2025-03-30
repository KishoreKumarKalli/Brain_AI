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

        # Calculate volumes for each label
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Count voxels for this label
            voxel_count = np.sum(segmentation == label)
            volume_mm3 = voxel_count * voxel_volume
            volume_ml = volume_mm3 / 1000  # Convert to mL

            # Get name for this class
            if label in tissue_names:
                class_name = tissue_names[label]
            else:
                class_name = f"Class {label}"

            # Store volume in dictionary
            volumes[f"{class_name}_volume_mm3"] = volume_mm3
            volumes[f"{class_name}_volume_ml"] = volume_ml

            # Add to total brain volume
            total_brain_volume += volume_mm3

        # Add total brain volume
        volumes["Total_Brain_Volume_mm3"] = total_brain_volume
        volumes["Total_Brain_Volume_ml"] = total_brain_volume / 1000

        return volumes

    def generate_pdf_subject_report(self, subject_id, original_image, segmentation_result,
                                     volumes, clinical_data, anomaly_map, diagnosis, filename):
        """
        Generate a PDF report for an individual subject.

        Args:
            subject_id (str): Subject identifier
            original_image (numpy.ndarray): Original MRI image
            segmentation_result (numpy.ndarray): Segmentation mask
            volumes (dict): Dictionary with volume measurements
            clinical_data (dict): Subject's clinical data
            anomaly_map (numpy.ndarray): Anomaly detection map
            diagnosis (str): Clinical diagnosis
            filename (str): Base filename for the report

        Returns:
            str: Path to the generated PDF file
        """
        # Create output path
        output_path = self.output_dir / f"{filename}.pdf"

        # Create PDF with multiple pages
        with PdfPages(output_path) as pdf:
            # Page 1: Overview and volumetric analysis
            plt.figure(figsize=(11.7, 8.3))  # A4 size in inches

            # Title
            plt.suptitle(f"Brain MRI Analysis Report: Subject {subject_id}", fontsize=16)
            plt.figtext(0.5, 0.92, f"Date: {self.report_time}", fontsize=10, ha="center")

            # Define grid layout
            gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 1])

            # Subject info section
            ax_info = plt.subplot(gs[0, :])
            info_text = f"Subject ID: {subject_id}\n"

            if diagnosis:
                info_text += f"Diagnosis: {diagnosis}\n"

            if isinstance(clinical_data, dict) and clinical_data:
                # Add selected clinical data (customize based on your data)
                key_clinical = ['MMSE', 'CDR', 'GDSCALE', 'Age', 'Sex', 'Education']
                for key in key_clinical:
                    if key in clinical_data:
                        info_text += f"{key}: {clinical_data[key]}\n"

            ax_info.text(0.05, 0.5, info_text, transform=ax_info.transAxes, fontsize=10,
                         verticalalignment='center', bbox=dict(boxstyle='round', alpha=0.1))
            ax_info.set_title("Subject Information", fontweight='bold')
            ax_info.axis('off')

            # Brain slices with segmentation overlay
            ax_slices = plt.subplot(gs[1, :2])

            # Find midpoint in the z-axis
            z_mid = original_image.shape[2] // 2

            # Select 3 slices around the middle
            slice_indices = [z_mid - 10, z_mid, z_mid + 10]
            slice_indices = [max(0, min(idx, original_image.shape[2] - 1)) for idx in slice_indices]

            # Create a composite image with the 3 slices
            composite_width = original_image.shape[1] * 3
            composite_height = original_image.shape[0]
            composite_img = np.zeros((composite_height, composite_width))
            composite_seg = np.zeros((composite_height, composite_width))

            for i, z in enumerate(slice_indices):
                start_col = i * original_image.shape[1]
                end_col = (i + 1) * original_image.shape[1]
                composite_img[:, start_col:end_col] = original_image[:, :, z]
                composite_seg[:, start_col:end_col] = segmentation_result[:, :, z]

            # Normalize the image for better visualization
            composite_img = (composite_img - composite_img.min()) / (composite_img.max() - composite_img.min())

            # Display the original image
            ax_slices.imshow(composite_img, cmap='gray')

            # Create custom colormap for segmentation
            cmap = plt.cm.get_cmap('jet', len(np.unique(segmentation_result)))

            # Overlay segmentation with transparency
            mask = composite_seg > 0  # Only overlay non-zero values
            overlay = np.ma.masked_where(~mask, composite_seg)
            ax_slices.imshow(overlay, cmap=cmap, alpha=0.5)

            ax_slices.set_title("Brain MRI with Segmentation Overlay", fontweight='bold')
            ax_slices.axis('off')

            # Volumetric analysis
            ax_volumes = plt.subplot(gs[1, 2])

            # Extract volume data for plotting
            structures = []
            vol_values = []

            for key, value in volumes.items():
                if "volume_ml" in key and "Total" not in key:
                    structure = key.replace("_volume_ml", "")
                    structures.append(structure)
                    vol_values.append(value)

            # Create horizontal bar chart
            y_pos = np.arange(len(structures))
            ax_volumes.barh(y_pos, vol_values, align='center')
            ax_volumes.set_yticks(y_pos)
            ax_volumes.set_yticklabels(structures)
            ax_volumes.set_xlabel('Volume (mL)')
            ax_volumes.set_title("Brain Structure Volumes", fontweight='bold')

            # Add volume table
            ax_vol_table = plt.subplot(gs[2, :2])
            ax_vol_table.axis('off')

            table_data = [["Brain Structure", "Volume (mL)", "% of Total Brain"]]
            total_vol = volumes.get("Total_Brain_Volume_ml", 0)

            for key, value in volumes.items():
                if "volume_ml" in key and "Total" not in key:
                    structure = key.replace("_volume_ml", "")
                    percentage = (value / total_vol * 100) if total_vol > 0 else 0
                    table_data.append([structure, f"{value:.2f}", f"{percentage:.2f}%"])

            # Add total row
            table_data.append(["Total Brain", f"{total_vol:.2f}", "100.00%"])

            table = ax_vol_table.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            for i in range(len(table_data)):
                if i == 0 or i == len(table_data) - 1:
                    for j in range(3):
                        table[(i, j)].set_facecolor('#E6F3FF')

            ax_vol_table.set_title("Volumetric Analysis", fontweight='bold')

            # Anomaly visualization (if available)
            ax_anomaly = plt.subplot(gs[2, 2])

            if anomaly_map is not None:
                # Display anomaly heatmap for the middle slice
                z_mid = anomaly_map.shape[2] // 2
                ax_anomaly.imshow(original_image[:, :, z_mid], cmap='gray')
                anomaly_display = ax_anomaly.imshow(anomaly_map[:, :, z_mid], cmap='hot',
                                                    alpha=0.7, vmin=0, vmax=1)
                plt.colorbar(anomaly_display, ax=ax_anomaly, fraction=0.046, pad=0.04)
                ax_anomaly.set_title("Anomaly Detection", fontweight='bold')
            else:
                ax_anomaly.text(0.5, 0.5, "Anomaly detection not available",
                                ha='center', va='center', transform=ax_anomaly.transAxes)
                ax_anomaly.set_title("Anomaly Detection", fontweight='bold')

            ax_anomaly.axis('off')

            # Add disclaimer at the bottom
            if self.config.get('reporting', {}).get('include_disclaimer', True):
                plt.figtext(0.5, 0.01, self.disclaimer, wrap=True,
                            horizontalalignment='center', fontsize=8, style='italic')

            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            pdf.savefig()
            plt.close()

            # Page 2: Clinical Data and 3D Visualization (if available)
            if clinical_data or anomaly_map is not None:
                plt.figure(figsize=(11.7, 8.3))  # A4 size in inches

                # Title
                plt.suptitle(f"Brain MRI Analysis Report: Subject {subject_id} - Detailed Analysis",
                             fontsize=16)

                # Grid layout
                gs = gridspec.GridSpec(2, 2)

                # Clinical data section (if available)
                ax_clinical = plt.subplot(gs[0, 0])

                if clinical_data:
                    # Create table from clinical data
                    clinical_table = []

                    # Header
                    clinical_table.append(["Clinical Measure", "Value"])

                    # Add relevant clinical data
                    exclude_keys = ['subject_id', 'PTID', 'Image_Path', 'path']

                    for key, value in clinical_data.items():
                        # Skip uninteresting keys
                        if key in exclude_keys or 'PATH' in key or 'ID' in key:
                            continue

                        # Format key for display
                        display_key = key.replace('_', ' ').title()
                        clinical_table.append([display_key, str(value)])

                    table = ax_clinical.table(cellText=clinical_table, loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.5)

                    for i in range(len(clinical_table)):
                        if i == 0:
                            for j in range(2):
                                table[(i, j)].set_facecolor('#E6F3FF')

                else:
                    ax_clinical.text(0.5, 0.5, "No clinical data available",
                                     ha='center', va='center', transform=ax_clinical.transAxes)

                ax_clinical.set_title("Clinical Assessment Data", fontweight='bold')
                ax_clinical.axis('off')

                # Coronal and sagittal views
                ax_coronal = plt.subplot(gs[0, 1])
                y_mid = original_image.shape[1] // 2
                coronal_slice = original_image[:, y_mid, :]
                coronal_seg = segmentation_result[:, y_mid, :]

                # Display the original image
                ax_coronal.imshow(coronal_slice.T, cmap='gray', origin='lower')

                # Overlay segmentation with transparency
                mask = coronal_seg.T > 0
                overlay = np.ma.masked_where(~mask, coronal_seg.T)
                ax_coronal.imshow(overlay, cmap=plt.cm.get_cmap('jet', len(np.unique(segmentation_result))),
                                  alpha=0.5, origin='lower')

                ax_coronal.set_title("Coronal View", fontweight='bold')
                ax_coronal.axis('off')

                # Visualization of 3D rendering (placeholder - in practice would use plotly)
                ax_render = plt.subplot(gs[1, :])
                ax_render.text(0.5, 0.5, "3D Rendering would be displayed here\n"
                                         "In a full implementation, a 3D surface rendering would be included using plotly or VTK.",
                               ha='center', va='center', transform=ax_render.transAxes,
                               bbox=dict(boxstyle='round', alpha=0.1))
                ax_render.set_title("3D Volume Rendering", fontweight='bold')
                ax_render.axis('off')

                # Add institution info
                plt.figtext(0.5, 0.02, f"Generated by {self.institution_name}",
                            horizontalalignment='center', fontsize=10)

                plt.tight_layout(rect=[0, 0.03, 1, 0.9])
                pdf.savefig()
                plt.close()

        print(f"PDF report generated: {output_path}")
        return str(output_path)

    def generate_html_subject_report(self, subject_id, original_image, segmentation_result,
                                      volumes, clinical_data, anomaly_map, diagnosis, filename):
        """
        Generate an HTML report for an individual subject.

        Args:
            subject_id (str): Subject identifier
            original_image (numpy.ndarray): Original MRI image
            segmentation_result (numpy.ndarray): Segmentation mask
            volumes (dict): Dictionary with volume measurements
            clinical_data (dict): Subject's clinical data
            anomaly_map (numpy.ndarray): Anomaly detection map
            diagnosis (str): Clinical diagnosis
            filename (str): Base filename for the report

        Returns:
            str: Path to the generated HTML file
        """
        # Create output path
        output_path = self.output_dir / f"{filename}.html"

        # Create folder for images if it doesn't exist
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Generate images for the HTML report
        image_paths = {}

        # Generate slices visualization
        plt.figure(figsize=(10, 4))
        z_indices = [original_image.shape[2] // 4, original_image.shape[2] // 2, 3 * original_image.shape[2] // 4]

        for i, z in enumerate(z_indices):
            plt.subplot(1, 3, i + 1)
            plt.imshow(original_image[:, :, z], cmap='gray')
            plt.imshow(segmentation_result[:, :, z], cmap='jet', alpha=0.5)
            plt.title(f"Slice {z}")
            plt.axis('off')

        slices_path = images_dir / f"{filename}_slices.png"
        plt.tight_layout()
        plt.savefig(slices_path, dpi=150)
        plt.close()
        image_paths['slices'] = f"images/{slices_path.name}"

        # Generate volumes bar chart
        plt.figure(figsize=(8, 6))
        structures = []
        vol_values = []

        for key, value in volumes.items():
            if "volume_ml" in key and "Total" not in key:
                structure = key.replace("_volume_ml", "")
                structures.append(structure)
                vol_values.append(value)

        y_pos = np.arange(len(structures))
        plt.barh(y_pos, vol_values, align='center')
        plt.yticks(y_pos, structures)
        plt.xlabel('Volume (mL)')
        plt.title("Brain Structure Volumes")

        volumes_path = images_dir / f"{filename}_volumes.png"
        plt.tight_layout()
        plt.savefig(volumes_path, dpi=150)
        plt.close()
        image_paths['volumes'] = f"images/{volumes_path.name}"

        # Generate anomaly visualization if available
        if anomaly_map is not None:
            plt.figure(figsize=(6, 6))
            z_mid = anomaly_map.shape[2] // 2
            plt.imshow(original_image[:, :, z_mid], cmap='gray')
            plt.imshow(anomaly_map[:, :, z_mid], cmap='hot', alpha=0.7, vmin=0, vmax=1)
            plt.colorbar(label='Anomaly Score')
            plt.title("Anomaly Detection")
            plt.axis('off')

            anomaly_path = images_dir / f"{filename}_anomaly.png"
            plt.tight_layout()
            plt.savefig(anomaly_path, dpi=150)
            plt.close()
            image_paths['anomaly'] = f"images/{anomaly_path.name}"

        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain MRI Analysis Report: {{ subject_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
                h1, h2, h3 { color: #0066cc; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background-color: #f5f5f5; padding: 20px; margin-bottom: 20px; border-bottom: 1px solid #ddd; }
                .section { margin-bottom: 30px; padding: 20px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                table, th, td { border: 1px solid #ddd; }
                th, td { padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
                .img-container { text-align: center; margin-bottom: 20px; }
                .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; border-top: 1px solid #ddd; padding-top: 20px; }
                .disclaimer { font-style: italic; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #ccc; margin-top: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Brain MRI Analysis Report</h1>
                    <p><strong>Subject ID:</strong> {{ subject_id }}</p>
                    <p><strong>Date:</strong> {{ report_date }}</p>
                    {% if diagnosis %}
                    <p><strong>Diagnosis:</strong> {{ diagnosis }}</p>
                    {% endif %}
                </div>

                <div class="section">
                    <h2>Brain MRI Visualization</h2>
                    <div class="img-container">
                        <img src="{{ images.slices }}" alt="Brain MRI Slices with Segmentation">
                        <p>Brain MRI with segmentation overlay (axial slices)</p>
                    </div>
                </div>

                <div class="section">
                    <h2>Volumetric Analysis</h2>
                    <div class="img-container">
                        <img src="{{ images.volumes }}" alt="Brain Structure Volumes">
                        <p>Volumetric measurements of brain structures</p>
                    </div>

                    <h3>Volume Measurements</h3>
                    <table>
                        <tr>
                            <th>Brain Structure</th>
                            <th>Volume (ml)</th>
                            <th>Volume (mm³)</th>
                            <th>% of Total Brain</th>
                        </tr>
                        {% for structure, values in volumes_table %}
                        <tr>
                            <td>{{ structure }}</td>
                            <td>{{ values.ml }}</td>
                            <td>{{ values.mm3 }}</td>
                            <td>{{ values.percentage }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>

                {% if clinical_data %}
                <div class="section">
                    <h2>Clinical Assessment Data</h2>
                    <table>
                        <tr>
                            <th>Measure</th>
                            <th>Value</th>
                        </tr>
                        {% for key, value in clinical_table %}
                        <tr>
                            <td>{{ key }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}

                {% if images.anomaly %}
                <div class="section">
                    <h2>Anomaly Detection</h2>
                    <div class="img-container">
                        <img src="{{ images.anomaly }}" alt="Anomaly Detection">
                        <p>Heatmap showing detected anomalies in brain tissue</p>
                    </div>
                </div>
                {% endif %}

                <div class="disclaimer">
                    {{ disclaimer }}
                </div>

                <div class="footer">
                    <p>Generated by {{ institution }} | {{ report_date }}</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Prepare template data
        template_data = {
            'subject_id': subject_id,
            'report_date': self.report_time,
            'diagnosis': diagnosis if diagnosis else None,
            'images': image_paths,
            'disclaimer': self.disclaimer,
            'institution': self.institution_name
        }

        # Prepare volumes table data
        volumes_table = []
        total_vol_mm3 = volumes.get("Total_Brain_Volume_mm3", 0)

        for key in volumes:
            if "volume_ml" in key and "Total" not in key:
                structure = key.replace("_volume_ml", "")
                vol_ml = volumes[key]
                vol_mm3 = volumes.get(f"{structure}_volume_mm3", 0)
                percentage = (vol_mm3 / total_vol_mm3 * 100) if total_vol_mm3 > 0 else 0

                volumes_table.append((
                    structure,
                    {
                        'ml': f"{vol_ml:.2f}",
                        'mm3': f"{vol_mm3:.2f}",
                        'percentage': f"{percentage:.2f}%"
                    }
                ))

        # Add total row
        volumes_table.append((
            "Total Brain",
            {
                'ml': f"{volumes.get('Total_Brain_Volume_ml', 0):.2f}",
                'mm3': f"{total_vol_mm3:.2f}",
                'percentage': "100.00%"
            }
        ))

        template_data['volumes_table'] = volumes_table

        # Prepare clinical data table
        clinical_table = []

        if clinical_data:
            exclude_keys = ['subject_id', 'PTID', 'Image_Path', 'path']

            for key, value in clinical_data.items():
                # Skip uninteresting keys
                if key in exclude_keys or 'PATH' in key or 'ID' in key:
                    continue

                # Format key for display
                display_key = key.replace('_', ' ').title()
                clinical_table.append((display_key, value))

        template_data['clinical_table'] = clinical_table

        # Render template
        template = jinja2.Template(html_template)
        html_content = template.render(**template_data)

        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report generated: {output_path}")
        return str(output_path)
