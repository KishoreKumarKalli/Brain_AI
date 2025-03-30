import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nibabel as nib
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind
import torch
import torchio as tio
from nilearn import plotting
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BrainVisualizer:
    """
    Class for visualizing brain MRI data, segmentation results, and statistical analyses.
    """

    def __init__(self, config, output_dir=None):
        """
        Initialize the visualizer.

        Args:
            config (dict): Configuration dictionary
            output_dir (str or Path, optional): Directory to save visualizations
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(config.get('results_dir', './results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default color maps for different tissue types
        self.tissue_cmap = {
            0: "gray",  # Background
            1: "Reds",  # Gray matter
            2: "Blues",  # White matter
            3: "Greens",  # CSF
            4: "Purples"  # Optional (ventricles or other specific structure)
        }

        # Labels for tissue types
        self.tissue_labels = {
            0: "Background",
            1: "Gray Matter",
            2: "White Matter",
            3: "CSF",
            4: "Other"
        }

        # Labels for diagnostic groups
        self.diagnosis_labels = {
            'CN': "Cognitive Normal",
            'MCI': "Mild Cognitive Impairment",
            'AD': "Alzheimer's Disease"
        }

        # Custom colormap for anomaly detection
        colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # Blue -> Cyan -> Green -> Yellow -> Red
        self.anomaly_cmap = LinearSegmentedColormap.from_list("anomaly_cmap", colors)

    def _create_custom_cmap(self, tissue_idx):
        """Create a custom colormap for overlay visualization."""
        if tissue_idx in self.tissue_cmap:
            return plt.cm.get_cmap(self.tissue_cmap[tissue_idx])
        return plt.cm.get_cmap("viridis")

    def visualize_slice(self, image_data, seg_data=None, slice_idx=None, axis=2,
                        alpha=0.5, save_path=None, anomaly_data=None):
        """
        Visualize a single slice of brain MRI with optional segmentation overlay.

        Args:
            image_data (np.ndarray): 3D brain MRI volume (X, Y, Z)
            seg_data (np.ndarray, optional): Segmentation mask with same dimensions
            slice_idx (int, optional): Slice index to visualize (default: middle slice)
            axis (int): Axis to slice along (0=sagittal, 1=coronal, 2=axial)
            alpha (float): Transparency level for segmentation overlay
            save_path (str, optional): Path to save the visualization
            anomaly_data (np.ndarray, optional): Anomaly score map
        """
        # Get dimensions and set default slice if needed
        if axis == 0:
            if slice_idx is None:
                slice_idx = image_data.shape[0] // 2
            image_slice = image_data[slice_idx, :, :]
            seg_slice = None if seg_data is None else seg_data[slice_idx, :, :]
            anomaly_slice = None if anomaly_data is None else anomaly_data[slice_idx, :, :]
        elif axis == 1:
            if slice_idx is None:
                slice_idx = image_data.shape[1] // 2
            image_slice = image_data[:, slice_idx, :]
            seg_slice = None if seg_data is None else seg_data[:, slice_idx, :]
            anomaly_slice = None if anomaly_data is None else anomaly_data[:, slice_idx, :]
        else:  # axis == 2
            if slice_idx is None:
                slice_idx = image_data.shape[2] // 2
            image_slice = image_data[:, :, slice_idx]
            seg_slice = None if seg_data is None else seg_data[:, :, slice_idx]
            anomaly_slice = None if anomaly_data is None else anomaly_data[:, :, slice_idx]

        # Set up figure
        if anomaly_data is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            ax1, ax2, ax3 = axes
        else:
            if seg_data is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            else:
                fig, ax1 = plt.subplots(figsize=(8, 8))
                ax2 = None

        # Plot original image
        im1 = ax1.imshow(image_slice, cmap='gray')
        ax1.set_title('Original MRI')
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)

        # Plot segmentation overlay if available
        if seg_slice is not None and ax2 is not None:
            # Create a masked array for each tissue type
            unique_labels = np.unique(seg_slice)
            overlay = np.zeros_like(image_slice)

            for label in unique_labels:
                if label == 0:  # Skip background
                    continue
                mask = seg_slice == label
                overlay[mask] = label

            # Display image with segmentation overlay
            im2 = ax2.imshow(image_slice, cmap='gray')
            im2_overlay = ax2.imshow(overlay, cmap='viridis', alpha=alpha)
            ax2.set_title('Segmentation Overlay')
            ax2.axis('off')
            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im2_overlay, cax=cax2)
            cbar.set_ticks(unique_labels)
            cbar.set_ticklabels([self.tissue_labels.get(label, f"Class {label}") for label in unique_labels])

        # Plot anomaly map if available
        if anomaly_slice is not None and ax3 is not None:
            im3 = ax3.imshow(anomaly_slice, cmap=self.anomaly_cmap)
            ax3.set_title('Anomaly Detection')
            ax3.axis('off')
            divider = make_axes_locatable(ax3)
            cax3 = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im3, cax=cax3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_3d_segmentation(self, image_data, seg_data, threshold=0.5,
                                  save_path=None, tissue_idx=1):
        """
        Create a 3D visualization of a brain segmentation for a specific tissue type.

        Args:
            image_data (np.ndarray): 3D brain MRI volume
            seg_data (np.ndarray): Segmentation mask
            threshold (float): Threshold for binary segmentation
            save_path (str, optional): Path to save the visualization
            tissue_idx (int): Index of the tissue to visualize
        """
        try:
            from mayavi import mlab

            # Create binary mask for the specified tissue
            tissue_mask = (seg_data == tissue_idx).astype(np.float32)

            # Create 3D visualization
            mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))

            # Add volume rendering of the original image
            src = mlab.pipeline.scalar_field(image_data)
            vol = mlab.pipeline.volume(src, vmin=np.percentile(image_data, 10),
                                       vmax=np.percentile(image_data, 90))
            vol.volume_property.opacity = 0.1

            # Add iso-surface for the segmentation
            src2 = mlab.pipeline.scalar_field(tissue_mask)
            contour = mlab.pipeline.iso_surface(src2, contours=[threshold], opacity=0.7,
                                                color=tuple(
                                                    np.array(plt.cm.get_cmap(self.tissue_cmap[tissue_idx])(0.7)[0:3])))

            # Set view properties
            mlab.view(azimuth=45, elevation=45, distance='auto')

            if save_path:
                mlab.savefig(save_path, size=(2400, 2400), magnification=3)
                mlab.close()
            else:
                mlab.show()

        except ImportError:
            print("Mayavi not installed. 3D visualization requires mayavi package.")
            print("Use 'pip install mayavi' to install it.")

            # Fall back to matplotlib 3D visualization (limited functionality)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Create binary mask for the specified tissue
            tissue_mask = (seg_data == tissue_idx).astype(np.float32)
            x, y, z = np.where(tissue_mask > threshold)

            # Downsample points for performance
            samples = min(10000, len(x))
            indices = np.random.choice(len(x), samples, replace=False)

            ax.scatter(x[indices], y[indices], z[indices], c='r', marker='o', alpha=0.05)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Visualization of {self.tissue_labels.get(tissue_idx, "Unknown")}')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def create_brain_animation(self, image_data, seg_data=None, axis=2, save_path=None, interval=50):
        """
        Create an animation of brain slices with optional segmentation overlay.

        Args:
            image_data (np.ndarray): 3D brain MRI volume
            seg_data (np.ndarray, optional): Segmentation mask
            axis (int): Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
            save_path (str): Path to save the animation (must end with .gif)
            interval (int): Time interval between frames in milliseconds
        """
        fig = plt.figure(figsize=(10, 10))

        if seg_data is not None:
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, 1, 1)

        # Determine number of slices based on axis
        if axis == 0:
            num_slices = image_data.shape[0]
        elif axis == 1:
            num_slices = image_data.shape[1]
        else:  # axis == 2
            num_slices = image_data.shape[2]

        # Function to update the figure for each frame
        def update_frame(i):
            ax.clear()

            # Get the slice for the current frame
            if axis == 0:
                image_slice = image_data[i, :, :]
                seg_slice = None if seg_data is None else seg_data[i, :, :]
            elif axis == 1:
                image_slice = image_data[:, i, :]
                seg_slice = None if seg_data is None else seg_data[:, i, :]
            else:  # axis == 2
                image_slice = image_data[:, :, i]
                seg_slice = None if seg_data is None else seg_data[:, :, i]

            # Display image
            ax.imshow(image_slice, cmap='gray')

            # Add segmentation overlay if available
            if seg_slice is not None:
                # Create a masked array for each tissue type
                unique_labels = np.unique(seg_slice)
                overlay = np.zeros_like(image_slice)

                for label in unique_labels:
                    if label == 0:  # Skip background
                        continue
                    mask = seg_slice == label
                    overlay[mask] = label

                ax.imshow(overlay, cmap='viridis', alpha=0.5)

            ax.set_title(f'Slice {i + 1}/{num_slices}')
            ax.axis('off')

        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=num_slices, interval=interval)

        # Save or display animation
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000 / interval)
            plt.close()
        else:
            plt.show()

    def plot_volume_differences(self, volumes_df, groupby='DX_GROUP', tissue_types=None,
                                save_path=None, normalized=False):
        """
        Plot volume differences between diagnostic groups.

        Args:
            volumes_df (pd.DataFrame): DataFrame with volume data and diagnostic groups
            groupby (str): Column name to group by (default: 'DX_GROUP')
            tissue_types (list): List of tissue types to include
            save_path (str, optional): Path to save the visualization
            normalized (bool): Whether to normalize volumes by total intracranial volume
        """
        # Set default tissue types if not provided
        if tissue_types is None:
            tissue_types = [f"Volume_{self.tissue_labels[i]}" for i in range(1, 5)]

        # Set figure size based on number of tissue types
        fig, axes = plt.subplots(1, len(tissue_types), figsize=(5 * len(tissue_types), 6))

        # Handle case with single tissue type
        if len(tissue_types) == 1:
            axes = [axes]

        # Normalize by TIV if requested
        if normalized:
            # Calculate total intracranial volume (sum of all tissue volumes)
            volumes_df['TIV'] = volumes_df[[col for col in volumes_df.columns if col.startswith('Volume_')]].sum(axis=1)

            # Normalize tissue volumes
            for tissue in tissue_types:
                volumes_df[f"{tissue}_norm"] = volumes_df[tissue] / volumes_df['TIV']

            # Update tissue types to use normalized values
            tissue_types = [f"{tissue}_norm" for tissue in tissue_types]

        # Plot each tissue type
        for i, tissue in enumerate(tissue_types):
            # Create boxplot
            sns.boxplot(x=groupby, y=tissue, data=volumes_df, palette='viridis', ax=axes[i])

            # Add individual data points
            sns.stripplot(x=groupby, y=tissue, data=volumes_df, color='black', alpha=0.5,
                          size=4, jitter=True, ax=axes[i])

            # Run t-tests between groups
            groups = volumes_df[groupby].unique()
            y_max = volumes_df[tissue].max() * 1.1
            y_min = volumes_df[tissue].min() * 0.9
            y_range = y_max - y_min

            for g1_idx, g1 in enumerate(groups[:-1]):
                for g2_idx, g2 in enumerate(groups[g1_idx + 1:], g1_idx + 1):
                    g1_data = volumes_df[volumes_df[groupby] == g1][tissue]
                    g2_data = volumes_df[volumes_df[groupby] == g2][tissue]

                    t_stat, p_val = ttest_ind(g1_data, g2_data)

                    # Add significance markers
                    sig_str = ""
                    if p_val < 0.001:
                        sig_str = "***"
                    elif p_val < 0.01:
                        sig_str = "**"
                    elif p_val < 0.05:
                        sig_str = "*"

                    if sig_str:
                        bar_y = y_max + (g2_idx - g1_idx) * y_range * 0.05
                        axes[i].plot([g1_idx, g2_idx], [bar_y, bar_y], 'k-')
                        axes[i].text((g1_idx + g2_idx) / 2, bar_y + y_range * 0.01,
                                     sig_str, ha='center', va='bottom')

            # Set titles and labels
            tissue_name = tissue.replace("Volume_", "").replace("_norm", "")
            axes[i].set_title(f"{tissue_name} Volume")
            axes[i].set_xlabel("")

            if normalized:
                axes[i].set_ylabel(f"Normalized {tissue_name} Volume")
            else:
                axes[i].set_ylabel(f"{tissue_name} Volume (mm³)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_clinical_correlations(self, volumetric_data, clinical_data, regions=None,
                                   clinical_measures=None, output_path=None, figsize=(18, 10)):
        """
        Create scatter plots showing correlations between brain region volumes and clinical measures.

        Args:
            volumetric_data (pandas.DataFrame): DataFrame with volumetric measurements
            clinical_data (pandas.DataFrame): DataFrame with clinical measurements
            regions (list, optional): List of brain regions to include in the analysis
            clinical_measures (list, optional): List of clinical measures to analyze
            output_path (str, optional): Path to save the output figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure with correlation plots
        """
        # Merge volumetric and clinical data using subject ID
        if 'subject_id' in volumetric_data.columns and 'subject_id' in clinical_data.columns:
            merged_data = pd.merge(volumetric_data, clinical_data, on='subject_id', how='inner')
        else:
            # Try other common ID columns if subject_id is not present
            common_id_cols = set(volumetric_data.columns).intersection(set(clinical_data.columns))
            id_candidates = [col for col in common_id_cols if any(id_term in col.lower()
                                                                  for id_term in ['id', 'subject', 'patient', 'ptid'])]

            if id_candidates:
                merged_data = pd.merge(volumetric_data, clinical_data, on=id_candidates[0], how='inner')
            else:
                raise ValueError("No common ID column found between volumetric and clinical data")

        # Identify volume columns if not specified
        if regions is None:
            regions = [col for col in merged_data.columns
                       if any(vol_term in col.lower() for vol_term in ['volume', 'vol', 'class_'])
                       and 'total' not in col.lower()]

        # Identify clinical measure columns if not specified
        if clinical_measures is None:
            # Common clinical assessment names in ADNI
            clinical_candidates = ['MMSE', 'ADAS', 'CDR', 'GDSCALE', 'FAQ', 'MOCA']
            clinical_measures = [col for col in merged_data.columns
                                 if any(measure in col for measure in clinical_candidates)]

        # Check if we have enough data to proceed
        if not regions or not clinical_measures:
            print("Warning: Not enough data columns identified for correlation analysis")
            return None

        # Number of plots to create
        n_plots = len(regions) * len(clinical_measures)
        if n_plots > 20:  # Limit the number of plots for readability
            print("Warning: Large number of plots requested. Limiting to top correlations.")

            # Calculate correlations to select the most significant ones
            correlation_strength = {}
            for region in regions:
                for measure in clinical_measures:
                    if pd.api.types.is_numeric_dtype(merged_data[region]) and pd.api.types.is_numeric_dtype(
                            merged_data[measure]):
                        corr = merged_data[[region, measure]].corr().iloc[0, 1]
                        correlation_strength[(region, measure)] = abs(corr)

            # Sort by correlation strength and take top 20
            top_correlations = sorted(correlation_strength.items(), key=lambda x: x[1], reverse=True)[:20]
            plot_pairs = [pair for pair, _ in top_correlations]
        else:
            plot_pairs = [(region, measure) for region in regions for measure in clinical_measures]

        # Calculate the grid size for the subplots
        n_cols = min(4, len(plot_pairs))
        n_rows = (len(plot_pairs) + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Flatten axes for easier indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()

        # Create scatter plots for each region-measure combination
        for i, (region, measure) in enumerate(plot_pairs):
            if i >= len(axes):  # Safety check
                break

            ax = axes[i]

            # Check if both columns are numeric
            if pd.api.types.is_numeric_dtype(merged_data[region]) and pd.api.types.is_numeric_dtype(
                    merged_data[measure]):
                # Create scatter plot
                scatter = ax.scatter(
                    merged_data[region],
                    merged_data[measure],
                    alpha=0.7,
                    s=50,
                    c='steelblue'
                )

                # Add regression line
                x = merged_data[region]
                y = merged_data[measure]

                # Remove NaNs
                mask = ~np.isnan(x) & ~np.isnan(y)
                x = x[mask]
                y = y[mask]

                if len(x) > 1:  # Need at least 2 points for regression
                    # Calculate regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    # Plot regression line
                    x_range = np.linspace(min(x), max(x), 100)
                    ax.plot(x_range, intercept + slope * x_range, 'r-', lw=2)

                    # Add correlation coefficient and p-value
                    ax.text(0.05, 0.95, f'r = {r_value:.2f}\np = {p_value:.3f}',
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                    # Improve region name formatting for title
                    region_name = region.replace('_volume_mm3', '').replace('class_', 'Class ').title()
                    ax.set_title(f'{region_name} vs. {measure}')
                    ax.set_xlabel(f'{region_name} (mm³)')
                    ax.set_ylabel(f'{measure} Score')

                    # If DX_GROUP is available, color points by diagnostic group
                    if 'DX_GROUP' in merged_data.columns:
                        groups = merged_data['DX_GROUP'].unique()
                        cmap = plt.cm.get_cmap('viridis', len(groups))

                        # Clear previous scatter
                        ax.clear()

                        # Plot each group with a different color
                        for j, group in enumerate(groups):
                            group_data = merged_data[merged_data['DX_GROUP'] == group]
                            ax.scatter(
                                group_data[region],
                                group_data[measure],
                                alpha=0.7,
                                s=50,
                                color=cmap(j),
                                label=group
                            )

                        # Add legend
                        if i % n_cols == n_cols - 1:  # Add legend only to rightmost plots
                            ax.legend(title="Diagnosis")

                        # Re-add regression line and correlation stats for all data
                        ax.plot(x_range, intercept + slope * x_range, 'r-', lw=2)
                        ax.text(0.05, 0.95, f'r = {r_value:.2f}\np = {p_value:.3f}',
                                transform=ax.transAxes, fontsize=10,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                        # Re-add titles
                        ax.set_title(f'{region_name} vs. {measure}')
                        ax.set_xlabel(f'{region_name} (mm³)')
                        ax.set_ylabel(f'{measure} Score')
                else:
                    ax.text(0.5, 0.5, "Insufficient data\nfor correlation",
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Non-numeric data",
                        ha='center', va='center', transform=ax.transAxes)

        # Hide any unused subplots
        for i in range(len(plot_pairs), len(axes)):
            axes[i].axis('off')

        # Main title
        fig.suptitle('Brain Region Volume vs. Clinical Measures Correlations', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Clinical correlations plot saved to: {output_path}")

        return fig