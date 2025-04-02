"""
Visualization module for brain segmentation analysis.
This module provides functionality for generating visualizations of
segmentation results, abnormality maps, and analysis data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import nibabel as nib
import logging
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('visualization')


class BrainVisualization:
    """
    Class for generating visualizations of brain data and analysis results.
    """

    def __init__(self, output_dir="./visualization_results"):
        """
        Initialize the visualization module.

        Args:
            output_dir (str): Directory to save visualization outputs
        """
        # Metadata
        self.visualization_date = "2025-04-02 14:57:16"
        self.created_by = "KishoreKumarKalli"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Default colormap configurations
        self.segmentation_cmap = {
            0: [0, 0, 0, 0],  # Background (transparent)
            1: [0.5, 0.5, 0.8, 1],  # Grey Matter
            2: [0.9, 0.9, 0.9, 1],  # White Matter
            3: [0, 0.7, 0.9, 1],  # CSF
            4: [0.8, 0.4, 0.2, 1],  # Cerebellum
            5: [0.2, 0.6, 0.5, 1],  # Ventricles
            6: [0.6, 0.2, 0.6, 1],  # Hippocampus
            7: [0.9, 0.7, 0.1, 1]  # Thalamus
        }

        # Structure labels
        self.structure_labels = {
            0: "Background",
            1: "Grey Matter",
            2: "White Matter",
            3: "CSF",
            4: "Cerebellum",
            5: "Ventricles",
            6: "Hippocampus",
            7: "Thalamus"
        }

        # Abnormality colormap (red-yellow)
        self.abnormality_cmap = cm.YlOrRd

        logger.info(f"Visualization module initialized at {self.visualization_date} by {self.created_by}")
        logger.info(f"Visualizations will be saved to {output_dir}")

    def create_segmentation_cmap(self, labels):
        """
        Create a colormap for segmentation visualization.

        Args:
            labels (list): List of label values in the segmentation

        Returns:
            matplotlib.colors.ListedColormap: Colormap for segmentation
        """
        # Get maximum label value
        max_label = max(max(labels), max(self.segmentation_cmap.keys()))

        # Create colors array
        colors = np.zeros((max_label + 1, 4))

        # Set colors for each label
        for label in range(max_label + 1):
            if label in self.segmentation_cmap:
                colors[label] = self.segmentation_cmap[label]
            else:
                # Generate a random color for unknown labels
                colors[label] = np.append(np.random.random(3), 1.0)

        # Set background to transparent
        colors[0] = [0, 0, 0, 0]

        return mcolors.ListedColormap(colors)

    def visualize_slice(self, image_data, seg_data=None, abnormality_map=None, slice_idx=None,
                        slice_axis=2, alpha=0.7, output_path=None, title=None):
        """
        Visualize a slice of brain image with optional segmentation overlay.

        Args:
            image_data (numpy.ndarray): 3D brain image
            seg_data (numpy.ndarray, optional): Segmentation mask
            abnormality_map (numpy.ndarray, optional): Abnormality map
            slice_idx (int, optional): Slice index (default: middle slice)
            slice_axis (int): Axis for slicing (0=sagittal, 1=coronal, 2=axial)
            alpha (float): Transparency for overlay
            output_path (str, optional): Path to save visualization
            title (str, optional): Custom title for the plot

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Determine slice index if not provided
        if slice_idx is None:
            slice_idx = image_data.shape[slice_axis] // 2

        # Get slice
        if slice_axis == 0:
            image_slice = image_data[slice_idx, :, :]
            seg_slice = None if seg_data is None else seg_data[slice_idx, :, :]
            abnorm_slice = None if abnormality_map is None else abnormality_map[slice_idx, :, :]
            view_name = "Sagittal"
        elif slice_axis == 1:
            image_slice = image_data[:, slice_idx, :]
            seg_slice = None if seg_data is None else seg_data[:, slice_idx, :]
            abnorm_slice = None if abnormality_map is None else abnormality_map[:, slice_idx, :]
            view_name = "Coronal"
        else:  # slice_axis == 2
            image_slice = image_data[:, :, slice_idx]
            seg_slice = None if seg_data is None else seg_data[:, :, slice_idx]
            abnorm_slice = None if abnormality_map is None else abnormality_map[:, :, slice_idx]
            view_name = "Axial"

        # Determine number of subplots needed
        n_plots = 1 + (seg_data is not None) + (abnormality_map is not None)

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

        # If there's only one plot, wrap it in a list for easier indexing
        if n_plots == 1:
            axes = [axes]

        # Plot image
        axes[0].imshow(image_slice.T, cmap='gray')
        axes[0].set_title(f"{view_name} View - Original Image (Slice {slice_idx})")
        axes[0].axis('off')

        # Plot segmentation if provided
        if seg_data is not None:
            plot_idx = 1

            # Create segmentation colormap
            if seg_slice is not None and np.any(seg_slice > 0):
                labels = np.unique(seg_slice).astype(int)
                cmap = self.create_segmentation_cmap(labels)

                # Plot segmentation overlay
                im = axes[plot_idx].imshow(image_slice.T, cmap='gray')
                seg = axes[plot_idx].imshow(seg_slice.T, cmap=cmap, alpha=alpha)
                axes[plot_idx].set_title(f"{view_name} View - Segmentation Overlay (Slice {slice_idx})")
                axes[plot_idx].axis('off')

                # Add a legend for segmentation
                legend_handles = []
                for label in labels:
                    if label > 0:  # Skip background
                        name = self.structure_labels.get(label, f"Label {label}")
                        color = cmap(label)
                        patch = plt.Rectangle((0, 0), 1, 1, fc=color)
                        legend_handles.append((patch, name))

                # Place legend outside and to the right of the plot
                axes[plot_idx].legend(*zip(*legend_handles), loc='upper left', bbox_to_anchor=(1.05, 1))

        # Plot abnormality map if provided
        if abnormality_map is not None:
            plot_idx = 2 if seg_data is not None else 1

            # Plot abnormality overlay
            im = axes[plot_idx].imshow(image_slice.T, cmap='gray')
            ab = axes[plot_idx].imshow(abnorm_slice.T, cmap=self.abnormality_cmap, alpha=alpha)
            axes[plot_idx].set_title(f"{view_name} View - Abnormality Map (Slice {slice_idx})")
            axes[plot_idx].axis('off')

            # Add colorbar
            plt.colorbar(ab, ax=axes[plot_idx], label='Abnormality Score')

        # Add overall title if provided
        if title:
            plt.suptitle(title, fontsize=16)
        else:
            plt.suptitle(f"Brain Visualization - {view_name} View\nGenerated: {self.visualization_date}", fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved slice visualization to {output_path}")

        return fig

    def visualize_3d_render(self, seg_data, structures=None, colors=None, transparent=False, output_path=None):
        """
        Create a 3D rendering of brain structures from segmentation.

        Args:
            seg_data (numpy.ndarray): Segmentation mask
            structures (list, optional): List of structure labels to render
            colors (dict, optional): Custom colors for structures {label: color}
            transparent (bool): Whether to make the rendering transparent
            output_path (str, optional): Path to save visualization

        Returns:
            plotly.graph_objects.Figure: Interactive 3D rendering
        """
        logger.info("Creating 3D rendering of brain structures")

        # If no structures specified, use all non-background structures
        if structures is None:
            structures = [label for label in np.unique(seg_data) if label > 0]

        logger.info(f"Rendering structures: {structures}")

        # Create an empty Plotly figure
        fig = go.Figure()

        # Process each structure
        for label in structures:
            structure_name = self.structure_labels.get(label, f"Structure {label}")
            logger.info(f"Processing structure: {structure_name} (label: {label})")

            # Select color
            if colors and label in colors:
                color = colors[label]
            elif label in self.segmentation_cmap:
                color_rgba = self.segmentation_cmap[label]
                color = f'rgba({color_rgba[0] * 255},{color_rgba[1] * 255},{color_rgba[2] * 255},{color_rgba[3]})'
            else:
                # Random color as fallback
                rgb = np.random.random(3) * 255
                color = f'rgb({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})'

            # Create binary mask and extract surface mesh
            try:
                verts, faces, _, _ = measure.marching_cubes(seg_data == label, level=0.5)

                # Reduce mesh complexity if too large (optional)
                if len(verts) > 100000:
                    logger.warning(
                        f"Mesh for {structure_name} is very large ({len(verts)} vertices). Visualization may be slow.")

                # Create mesh for this structure
                x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
                i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

                # Add mesh to the figure
                opacity = 0.7 if transparent else 1.0
                fig.add_trace(
                    go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=structure_name))

            except Exception as e:
                logger.error(f"Error creating mesh for structure {structure_name}: {str(e)}")
                continue
        # Update layout
        fig.update_layout(
            title=f"3D Brain Structure Visualization<br>Generated: 2025-04-02 14:59:01 by KishoreKumarKalli",
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0, y=1)
        )

        # Save if output path is provided
        if output_path:
            # Different extensions for different outputs
            if output_path.endswith('.html'):
                fig.write_html(output_path)
                logger.info(f"Saved interactive 3D rendering to {output_path}")
            else:
                fig.write_image(output_path, width=1200, height=1000)
                logger.info(f"Saved static 3D rendering to {output_path}")

        return fig

    def create_multiview_visualization(self, image_data, seg_data=None, abnormality_map=None, output_path=None):
        """
        Create a multi-view visualization with axial, sagittal and coronal views.

        Args:
            image_data (numpy.ndarray): 3D brain image
            seg_data (numpy.ndarray, optional): Segmentation mask
            abnormality_map (numpy.ndarray, optional): Abnormality map
            output_path (str, optional): Path to save visualization

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Creating multi-view visualization")

        # Get middle slice indices
        x_mid = image_data.shape[0] // 2
        y_mid = image_data.shape[1] // 2
        z_mid = image_data.shape[2] // 2

        # Create figure with grid layout
        fig = plt.figure(figsize=(20, 15))
        grid = GridSpec(2, 3, figure=fig)

        # Prepare for segmentation overlay
        if seg_data is not None:
            # Get all labels
            labels = np.unique(seg_data).astype(int)
            seg_cmap = self.create_segmentation_cmap(labels)

            # Create legend items
            legend_handles = []
            for label in labels:
                if label > 0:  # Skip background
                    name = self.structure_labels.get(label, f"Label {label}")
                    color = seg_cmap(label)
                    patch = plt.Rectangle((0, 0), 1, 1, fc=color)
                    legend_handles.append((patch, name))

        # Axial view (top row, left)
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.imshow(image_data[:, :, z_mid], cmap='gray')
        if seg_data is not None:
            ax1.imshow(seg_data[:, :, z_mid], cmap=seg_cmap, alpha=0.7)
        ax1.set_title(f"Axial View (z={z_mid})")
        ax1.axis('off')

        # Coronal view (top row, center)
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.imshow(image_data[:, y_mid, :].T, cmap='gray', origin='lower')
        if seg_data is not None:
            ax2.imshow(seg_data[:, y_mid, :].T, cmap=seg_cmap, alpha=0.7, origin='lower')
        ax2.set_title(f"Coronal View (y={y_mid})")
        ax2.axis('off')

        # Sagittal view (top row, right)
        ax3 = fig.add_subplot(grid[0, 2])
        ax3.imshow(image_data[x_mid, :, :].T, cmap='gray', origin='lower')
        if seg_data is not None:
            ax3.imshow(seg_data[x_mid, :, :].T, cmap=seg_cmap, alpha=0.7, origin='lower')
        ax3.set_title(f"Sagittal View (x={x_mid})")
        ax3.axis('off')

        # Abnormality map view (if provided)
        if abnormality_map is not None:
            # Get maximum abnormality value for consistent colormap scaling
            vmax = np.max(abnormality_map)
            if vmax <= 0:
                vmax = 1.0  # Fallback if abnormality map is all zeros

            # Axial abnormality view (bottom row, left)
            ax4 = fig.add_subplot(grid[1, 0])
            ax4.imshow(image_data[:, :, z_mid], cmap='gray')
            im4 = ax4.imshow(abnormality_map[:, :, z_mid], cmap=self.abnormality_cmap, alpha=0.7, vmin=0, vmax=vmax)
            ax4.set_title(f"Axial Abnormality (z={z_mid})")
            ax4.axis('off')

            # Coronal abnormality view (bottom row, center)
            ax5 = fig.add_subplot(grid[1, 1])
            ax5.imshow(image_data[:, y_mid, :].T, cmap='gray', origin='lower')
            ax5.imshow(abnormality_map[:, y_mid, :].T, cmap=self.abnormality_cmap, alpha=0.7, vmin=0, vmax=vmax,
                       origin='lower')
            ax5.set_title(f"Coronal Abnormality (y={y_mid})")
            ax5.axis('off')

            # Sagittal abnormality view (bottom row, right)
            ax6 = fig.add_subplot(grid[1, 2])
            ax6.imshow(image_data[x_mid, :, :].T, cmap='gray', origin='lower')
            ax6.imshow(abnormality_map[x_mid, :, :].T, cmap=self.abnormality_cmap, alpha=0.7, vmin=0, vmax=vmax,
                       origin='lower')
            ax6.set_title(f"Sagittal Abnormality (x={x_mid})")
            ax6.axis('off')

            # Add colorbar for abnormality map
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
            cbar = plt.colorbar(im4, cax=cbar_ax)
            cbar.set_label('Abnormality Score')

        # Add segmentation legend if applicable
        if seg_data is not None:
            fig.legend(*zip(*legend_handles), loc='lower center', ncol=min(4, len(legend_handles)),
                       bbox_to_anchor=(0.5, 0.02))

        # Add title
        plt.suptitle(f"Multi-View Brain Visualization\nGenerated: 2025-04-02 14:59:01 by KishoreKumarKalli",
                     fontsize=16)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 0.9 if abnormality_map is not None else 1, 0.95])

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-view visualization to {output_path}")

        return fig

    def create_segmentation_comparison(self, image_data, seg_data1, seg_data2, output_path=None, title=None):
        """
        Create a visualization comparing two segmentations.

        Args:
            image_data (numpy.ndarray): 3D brain image
            seg_data1 (numpy.ndarray): First segmentation mask
            seg_data2 (numpy.ndarray): Second segmentation mask
            output_path (str, optional): Path to save visualization
            title (str, optional): Custom title for the comparison

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Creating segmentation comparison visualization")

        # Get middle slice indices
        z_mid = image_data.shape[2] // 2

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Plot original image with first segmentation
        axes[0].imshow(image_data[:, :, z_mid], cmap='gray')

        # Create segmentation colormaps
        labels1 = np.unique(seg_data1).astype(int)
        cmap1 = self.create_segmentation_cmap(labels1)

        labels2 = np.unique(seg_data2).astype(int)
        cmap2 = self.create_segmentation_cmap(labels2)

        # Plot first segmentation
        axes[0].imshow(seg_data1[:, :, z_mid], cmap=cmap1, alpha=0.7)
        axes[0].set_title("Segmentation 1")
        axes[0].axis('off')

        # Plot original image with second segmentation
        axes[1].imshow(image_data[:, :, z_mid], cmap='gray')
        axes[1].imshow(seg_data2[:, :, z_mid], cmap=cmap2, alpha=0.7)
        axes[1].set_title("Segmentation 2")
        axes[1].axis('off')

        # Create difference mask
        diff_mask = seg_data1[:, :, z_mid] != seg_data2[:, :, z_mid]

        # Plot original image with differences highlighted
        axes[2].imshow(image_data[:, :, z_mid], cmap='gray')
        axes[2].imshow(diff_mask, cmap='hot', alpha=0.7)
        axes[2].set_title("Differences")
        axes[2].axis('off')

        # Add legend for each segmentation
        legend_handles1 = []
        for label in labels1:
            if label > 0:  # Skip background
                name = self.structure_labels.get(label, f"Label {label}")
                color = cmap1(label)
                patch = plt.Rectangle((0, 0), 1, 1, fc=color)
                legend_handles1.append((patch, name))

        if legend_handles1:
            axes[0].legend(*zip(*legend_handles1), loc='upper left', bbox_to_anchor=(1.05, 1))

        legend_handles2 = []
        for label in labels2:
            if label > 0:  # Skip background
                name = self.structure_labels.get(label, f"Label {label}")
                color = cmap2(label)
                patch = plt.Rectangle((0, 0), 1, 1, fc=color)
                legend_handles2.append((patch, name))

        if legend_handles2:
            axes[1].legend(*zip(*legend_handles2), loc='upper left', bbox_to_anchor=(1.05, 1))

        # Add percentage difference info
        total_voxels = np.prod(diff_mask.shape)
        diff_voxels = np.sum(diff_mask)
        diff_percentage = (diff_voxels / total_voxels) * 100

        diff_text = f"Difference: {diff_voxels} voxels ({diff_percentage:.2f}%)"
        axes[2].text(0.5, -0.1, diff_text, transform=axes[2].transAxes,
                     ha='center', va='center', fontsize=12)

        # Add overall title
        if title:
            plt.suptitle(title, fontsize=16)
        else:
            plt.suptitle(
                f"Segmentation Comparison (Axial Slice {z_mid})\nGenerated: 2025-04-02 14:59:01 by KishoreKumarKalli",
                fontsize=14)

        # Adjust layout
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved segmentation comparison to {output_path}")

        return fig

    def create_volume_comparison_chart(self, volumes_df, structures=None, subjects=None,
                                       normalize_by_icv=False, output_path=None):
        """
        Create chart comparing volumes of brain structures across subjects.

        Args:
            volumes_df (pandas.DataFrame): DataFrame with volume data
            structures (list, optional): List of structures to include
            subjects (list, optional): List of subjects to include
            normalize_by_icv (bool): Whether to normalize volumes by ICV
            output_path (str, optional): Path to save visualization

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Creating volume comparison chart")

        # Filter by structures if specified
        if structures:
            volumes_df = volumes_df[volumes_df['structure'].isin(structures)]

        # Filter by subjects if specified
        if subjects:
            volumes_df = volumes_df[volumes_df['subject_id'].isin(subjects)]

        # Check if we have data to plot
        if volumes_df.empty:
            logger.warning("No data available for volume comparison chart")
            return None

        # Determine which volume column to use
        volume_col = 'volume_percentage_of_icv' if normalize_by_icv else 'volume_ml'

        # Check if the column exists
        if volume_col not in volumes_df.columns:
            if volume_col == 'volume_percentage_of_icv':
                logger.warning("ICV-normalized volumes not available. Using absolute volumes.")
                volume_col = 'volume_ml'
            else:
                logger.error(f"Volume column '{volume_col}' not found in data")
                return None

        # Create bar chart
        plt.figure(figsize=(14, 8))
        chart = sns.barplot(x='structure', y=volume_col, hue='subject_id', data=volumes_df)

        # Customize appearance
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
        plt.ylabel('Volume (% of ICV)' if normalize_by_icv else 'Volume (ml)')
        plt.xlabel('Brain Structure')
        plt.title(f"Brain Structure Volume Comparison\nGenerated: 2025-04-02 14:59:01 by KishoreKumarKalli",
                  fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Subject ID')

        # Tight layout to avoid clipping
        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved volume comparison chart to {output_path}")

        return plt.gcf()

    def create_interactive_dashboard(self, image_data, seg_data=None, abnormality_map=None,
                                     clinical_data=None, output_path=None):
        """
        Create an interactive HTML dashboard visualization.

        Args:
            image_data (numpy.ndarray): 3D brain image
            seg_data (numpy.ndarray, optional): Segmentation mask
            abnormality_map (numpy.ndarray, optional): Abnormality map
            clinical_data (pandas.DataFrame, optional): Clinical data
            output_path (str): Path to save the HTML dashboard

        Returns:
            str: Path to the saved dashboard
        """
        logger.info("Creating interactive dashboard")

        if not output_path:
            output_path = os.path.join(self.output_dir, "interactive_dashboard.html")

        # Create HTML components
        html_parts = []

        # Add header
        html_parts.append(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .visualization-section {{ margin-bottom: 40px; }}
                .row {{ display: flex; flex-wrap: wrap; margin: 0 -15px; }}
                .column {{ flex: 50%; padding: 0 15px; box-sizing: border-box; }}
                h1, h2, h3 {{ color: #444; }}
                .timestamp {{ color: #666; font-size: 14px; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Brain Analysis Dashboard</h1>
                    <p class="timestamp">Generated: 2025-04-02 14:59:01 by KishoreKumarKalli</p>
                </div>
        """)

        # Create multiview visualization
        logger.info("Creating multiview visualization for dashboard")
        multiview_fig = self.create_multiview_visualization(
            image_data, seg_data, abnormality_map)

        # Save as temporary file
        multiview_path = os.path.join(self.output_dir, "temp_multiview.png")
        multiview_fig.savefig(multiview_path, dpi=150, bbox_inches='tight')
        plt.close(multiview_fig)

        # Add multiview section
        html_parts.append(f"""
                <div class="visualization-section">
                    <h2>Multi-View Brain Visualization</h2>
                    <img src="temp_multiview.png" alt="Multi-view visualization">
                </div>
        """)

        # Add 3D visualization if segmentation is available
        if seg_data is not None:
            logger.info("Creating 3D visualization for dashboard")
            try:
                # Create 3D visualization
                structures = [label for label in np.unique(seg_data) if label > 0][:4]  # Limit to first 4 structures
                fig_3d = self.visualize_3d_render(seg_data, structures=structures)

                # Convert to HTML
                html_3d = fig_3d.to_html(full_html=False)

                # Add 3D section
                html_parts.append(f"""
                    <div class="visualization-section">
                        <h2>3D Structure Visualization</h2>
                        {html_3d}
                    </div>
                """)
            except Exception as e:
                logger.error(f"Error creating 3D visualization: {str(e)}")

        # Add clinical data section if available
        if clinical_data is not None and not clinical_data.empty:
            logger.info("Adding clinical data to dashboard")
            try:
                # Create a summary table
                html_table = clinical_data.to_html(classes='table table-striped', index=False)

                # Add clinical data section
                html_parts.append(f"""
                    <div class="visualization-section">
                        <h2>Clinical Data Summary</h2>
                        {html_table}
                    </div>
                """)
            except Exception as e:
                logger.error(f"Error adding clinical data: {str(e)}")

        # Add footer and close tags
        html_parts.append("""
            </div>
        </body>
        </html>
        """)

        # Combine all HTML parts
        dashboard_html = "\n".join(html_parts)

        # Save dashboard
        with open(output_path, 'w') as f:
            f.write(dashboard_html)

        logger.info(f"Interactive dashboard saved to {output_path}")

        return output_path