import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from skimage import measure
from scipy.ndimage import zoom
import scipy.stats as stats

# Define custom colormaps for different tissue classes
TISSUE_COLORS = {
    0: [0, 0, 0, 0],  # Background - transparent
    1: [0.7, 0.3, 0.3, 1],  # Gray matter - red
    2: [0.3, 0.7, 0.3, 1],  # White matter - green
    3: [0.3, 0.3, 0.7, 1],  # CSF/Ventricles - blue
    4: [0.7, 0.7, 0.3, 1],  # Hippocampus - yellow
    5: [0.7, 0.3, 0.7, 1],  # Tumor/Lesion - purple
}

# Create a dict mapping class index to name
TISSUE_NAMES = {
    0: "Background",
    1: "Gray Matter",
    2: "White Matter",
    3: "CSF/Ventricles",
    4: "Hippocampus",
    5: "Tumor/Lesion"
}


# Create a custom colormap for segmentation visualization
def create_segmentation_cmap(num_classes=6):
    """Create a custom colormap for segmentation visualization."""
    colors = []
    for i in range(num_classes):
        if i in TISSUE_COLORS:
            colors.append(TISSUE_COLORS[i])
        else:
            colors.append([0.5, 0.5, 0.5, 1])  # Default gray for undefined classes

    return LinearSegmentedColormap.from_list("segmentation", colors, N=num_classes)


class BrainMRIVisualizer:
    """
    Class for visualizing brain MRI data and segmentation results.

    Args:
        num_classes (int): Number of tissue classes in the segmentation
    """

    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.cmap = create_segmentation_cmap(num_classes)

    def plot_slices(self, image, segmentation=None, num_slices=5, title=None,
                    alpha=0.5, output_path=None, figsize=(15, 8)):
        """
        Plot multiple slices of brain MRI with optional segmentation overlay.

        Args:
            image (numpy.ndarray): 3D brain MRI volume
            segmentation (numpy.ndarray, optional): Segmentation mask
            num_slices (int): Number of slices to show
            title (str): Title for the plot
            alpha (float): Transparency of segmentation overlay
            output_path (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Make sure image is 3D
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        if segmentation is not None and segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        # Calculate slice indices
        z_indices = np.linspace(0, image.shape[2] - 1, num_slices + 2)[1:-1].astype(int)

        # Create figure
        fig, axes = plt.subplots(1, num_slices, figsize=figsize)

        # Set title
        if title:
            fig.suptitle(title, fontsize=16)

        # Plot each slice
        for i, z in enumerate(z_indices):
            ax = axes[i]

            # Plot original image
            ax.imshow(image[:, :, z], cmap='gray')

            # Overlay segmentation if provided
            if segmentation is not None:
                ax.imshow(segmentation[:, :, z], cmap=self.cmap, alpha=alpha, vmin=0, vmax=self.num_classes - 1)

            ax.set_title(f'Slice {z}')
            ax.axis('off')

        plt.tight_layout()

        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {output_path}")

        return fig

    def plot_3d_surface(self, segmentation, class_idx=1, threshold=0.5,
                        output_path=None, smoothing=1, opacity=0.7):
        """
        Create a 3D surface rendering of a segmentation class.

        Args:
            segmentation (numpy.ndarray): 3D segmentation mask
            class_idx (int): Class index to visualize
            threshold (float): Threshold for surface generation
            output_path (str, optional): Path to save the figure
            smoothing (int): Smoothing factor for the surface
            opacity (float): Opacity of the surface

        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Make sure segmentation is 3D
        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        # Extract the class mask
        mask = (segmentation == class_idx).astype(float)

        # Downsample for performance if needed
        if max(mask.shape) > 128:
            zoom_factors = [128 / s for s in mask.shape]
            mask = zoom(mask, zoom_factors, order=1)

        # Apply smoothing if needed
        if smoothing > 1:
            from scipy.ndimage import gaussian_filter
            mask = gaussian_filter(mask, sigma=smoothing)

        # Extract vertices and faces using marching cubes
        verts, faces, _, _ = measure.marching_cubes(mask, threshold)

        # Create mesh3d trace
        x, y, z = verts.T
        i, j, k = faces.T

        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=opacity,
                colorscale=[[0, 'rgb({},{},{})'.format(*np.array(TISSUE_COLORS[class_idx][:3]) * 255)],
                            [1, 'rgb({},{},{})'.format(*np.array(TISSUE_COLORS[class_idx][:3]) * 255)]],
                intensity=np.ones(len(x)),
                intensitymode='cell',
                showscale=False,
                name=TISSUE_NAMES.get(class_idx, f"Class {class_idx}")
            )
        ])

        # Set layout
        fig.update_layout(
            title=f"3D Surface Rendering - {TISSUE_NAMES.get(class_idx, f'Class {class_idx}')}",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Save figure if output path is provided
        if output_path:
            fig.write_html(output_path)
            print(f"3D visualization saved to {output_path}")

        return fig

    def plot_multi_class_rendering(self, segmentation, classes=None,
                                   output_path=None, smoothing=1):
        """
        Create a multi-class 3D rendering with different colors.

        Args:
            segmentation (numpy.ndarray): 3D segmentation mask
            classes (list): List of class indices to visualize
            output_path (str, optional): Path to save the figure
            smoothing (int): Smoothing factor for the surfaces

        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Make sure segmentation is 3D
        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        # Determine classes to visualize
        if classes is None:
            classes = list(range(1, self.num_classes))  # Skip background class 0

        # Downsample for performance if needed
        downsampled_seg = segmentation
        if max(segmentation.shape) > 128:
            zoom_factors = [128 / s for s in segmentation.shape]
            downsampled_seg = zoom(segmentation, zoom_factors, order=0)

        # Create figure
        fig = go.Figure()

        # Add each class as a separate surface
        for class_idx in classes:
            if class_idx == 0:  # Skip background
                continue

            # Extract the class mask
            mask = (downsampled_seg == class_idx).astype(float)

            # Apply smoothing if needed
            if smoothing > 1:
                from scipy.ndimage import gaussian_filter
                mask = gaussian_filter(mask, sigma=smoothing)

            # Skip if class is not present
            if mask.max() < 0.5:
                continue

            # Extract vertices and faces using marching cubes
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, 0.5)
            except:
                print(f"Warning: Could not generate surface for class {class_idx}")
                continue

            # Create mesh3d trace
            x, y, z = verts.T
            i, j, k = faces.T

            # Get color for this class
            color_rgb = TISSUE_COLORS.get(class_idx, [0.5, 0.5, 0.5])[:3]
            color_str = 'rgb({},{},{})'.format(*(np.array(color_rgb) * 255).astype(int))

            # Add mesh to figure
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    opacity=0.7,
                    color=color_str,
                    name=TISSUE_NAMES.get(class_idx, f"Class {class_idx}")
                )
            )

        # Set layout
        fig.update_layout(
            title="Multi-Class 3D Rendering",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Save figure if output path is provided
        if output_path:
            fig.write_html(output_path)
            print(f"3D visualization saved to {output_path}")

        return fig

    def create_volume_comparison(self, volumes_df, class_columns=None, group_column=None,
                                 output_path=None, figsize=(12, 8)):
        """
        Create volumetric comparison between groups.

        Args:
            volumes_df (pandas.DataFrame): DataFrame with volumetric data
            class_columns (list): List of column names with volume data
            group_column (str): Column name for grouping
            output_path (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Determine columns to plot
        if class_columns is None:
            class_columns = [col for col in volumes_df.columns if "volume" in col.lower()]

        # Create figure
        fig, axes = plt.subplots(1, len(class_columns), figsize=figsize)

        # Single column case
        if len(class_columns) == 1:
            axes = [axes]

        # Plot each class
        for i, col in enumerate(class_columns):
            ax = axes[i]

            # Format column name for title
            title = col.replace("_volume_mm3", "").replace("class_", "Class ").title()

            # Create box plot or violin plot based on group presence
            if group_column and group_column in volumes_df.columns:
                sns.boxplot(x=group_column, y=col, data=volumes_df, ax=ax)

                # Add statistical test if more than one group
                groups = volumes_df[group_column].unique()
                if len(groups) > 1:
                    # Perform ANOVA if more than 2 groups, t-test otherwise
                    if len(groups) > 2:
                        groups_data = [volumes_df[volumes_df[group_column] == g][col].values
                                       for g in groups]
                        f_val, p_val = stats.f_oneway(*groups_data)
                        test_name = "ANOVA"
                    else:
                        g1 = volumes_df[volumes_df[group_column] == groups[0]][col].values
                        g2 = volumes_df[volumes_df[group_column] == groups[1]][col].values
                        _, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                        test_name = "t-test"

                    # Add p-value annotation
                    ax.text(0.5, 0.95, f"{test_name} p={p_val:.3f}",
                            ha='center', va='top', transform=ax.transAxes)
            else:
                sns.boxplot(y=col, data=volumes_df, ax=ax)

            ax.set_title(title)
            ax.set_ylabel("Volume (mm³)")

        plt.tight_layout()

        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {output_path}")

        return fig

    def plot_correlation_matrix(self, data_df, columns=None, clinical_columns=None,
                                output_path=None, figsize=(10, 8)):
        """
        Create correlation matrix between volumetric and clinical variables.

        Args:
            data_df (pandas.DataFrame): DataFrame with both volumetric and clinical data
            columns (list): List of column names to include
            clinical_columns (list): List of clinical column names
            output_path (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Determine columns to plot
        if columns is None:
            vol_columns = [col for col in data_df.columns if "volume" in col.lower()]
            if clinical_columns is None:
                # Try to identify clinical columns (not including ID, volume, or timestamp columns)
                exclude_patterns = ['id', 'volume', 'date', 'time', 'path']
                clinical_columns = [col for col in data_df.columns
                                    if not any(pattern in col.lower() for pattern in exclude_patterns)
                                    and col not in vol_columns]

            columns = vol_columns + clinical_columns

        # Only include numeric columns
        numeric_columns = data_df[columns].select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) < 2:
            print("Warning: Not enough numeric columns for correlation analysis")
            return None

        # Calculate correlation matrix
        corr_matrix = data_df[numeric_columns].corr()

        # Create figure
        plt.figure(figsize=figsize)

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})

        plt.title("Correlation Matrix", fontsize=16)

        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {output_path}")

        return plt.gcf()

    def create_anomaly_visualization(self, original_image, segmentation, anomaly_score,
                                     slices=None, threshold=0.7, output_path=None, figsize=(15, 10)):
        """
        Create visualization of anomaly detection results.

        Args:
            original_image (numpy.ndarray): Original 3D brain image
            segmentation (numpy.ndarray): Segmentation result
            anomaly_score (numpy.ndarray): Anomaly score map
            slices (list): List of slice indices to show
            threshold (float): Threshold for binary anomaly detection
            output_path (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        # Make sure inputs are 3D
        if original_image.ndim == 4 and original_image.shape[0] == 1:
            original_image = original_image[0]

        if segmentation.ndim == 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

        if anomaly_score.ndim == 4 and anomaly_score.shape[0] == 1:
            anomaly_score = anomaly_score[0]

        # Determine slices to show
        if slices is None:
            # Find slices with highest anomaly scores
            slice_scores = [anomaly_score[:, :, i].max() for i in range(anomaly_score.shape[2])]
            top_slices = np.argsort(slice_scores)[-4:]  # Top 4 slices
            slices = sorted(top_slices)

        # Create binary anomaly mask
        anomaly_binary = anomaly_score > threshold

        # Create figure
        num_slices = len(slices)
        fig, axes = plt.subplots(3, num_slices, figsize=figsize)

        # Plot each slice
        for i, slice_idx in enumerate(slices):
            # Original image
            axes[0, i].imshow(original_image[:, :, slice_idx], cmap='gray')
            axes[0, i].set_title(f'Slice {slice_idx}')
            axes[0, i].axis('off')

            # Segmentation overlay
            axes[1, i].imshow(original_image[:, :, slice_idx], cmap='gray')
            axes[1, i].imshow(segmentation[:, :, slice_idx], cmap=self.cmap, alpha=0.7)
            axes[1, i].set_title('Segmentation')
            axes[1, i].axis('off')

            # Anomaly heatmap overlay
            axes[2, i].imshow(original_image[:, :, slice_idx], cmap='gray')
            anomaly_map = axes[2, i].imshow(anomaly_score[:, :, slice_idx],
                                            cmap='hot', alpha=0.7, vmin=0, vmax=1)
            axes[2, i].contour(anomaly_binary[:, :, slice_idx],
                               colors='red', levels=[0.5], linewidths=1)
            axes[2, i].set_title('Anomaly Score')
            axes[2, i].axis('off')

        # Add colorbar for anomaly score
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        fig.colorbar(anomaly_map, cax=cbar_ax)
        cbar_ax.set_ylabel('Anomaly Score', rotation=270, labelpad=15)

        plt.suptitle('Anomaly Detection Results', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {output_path}")

        return fig

    def generate_subject_report(subject_id, segmentation, original_image, clinical_data=None, anomaly_score=None,
                                output_dir=None):
        """
        Generate a comprehensive report for a single subject.

        Args:
            subject_id (str): Subject ID
            segmentation (numpy.ndarray): Segmentation map (3D)
            original_image (numpy.ndarray): Original MRI image (3D)
            clinical_data (dict, optional): Dictionary of clinical data
            anomaly_score (numpy.ndarray, optional): Anomaly score map (3D)
            output_dir (str, optional): Directory to save the report

        Returns:
            matplotlib.figure.Figure: Report figure
        """
        # Create figure
        fig = plt.figure(figsize=(20, 15))

        # Main title with subject ID and date
        current_date = datetime.now().strftime("%Y-%m-%d")
        fig.suptitle(f"Brain MRI Analysis Report: Subject {subject_id} | {current_date}", fontsize=20)

        # Get midpoints for each axis for visualization
        x_mid, y_mid, z_mid = [dim // 2 for dim in original_image.shape]

        # Plot original MRI slices from three orthogonal planes
        axial_slice = original_image[:, :, z_mid]
        coronal_slice = original_image[:, y_mid, :]
        sagittal_slice = original_image[x_mid, :, :]

        # Plot segmentation slices from three orthogonal planes
        seg_axial = segmentation[:, :, z_mid]
        seg_coronal = segmentation[:, y_mid, :]
        seg_sagittal = segmentation[x_mid, :, :]

        # Define grid spec for complex layout
        gs = gridspec.GridSpec(4, 6)

        # Axial view (top down)
        ax1 = plt.subplot(gs[0, 0:2])
        ax1.imshow(axial_slice, cmap='gray')
        ax1.set_title("Axial Slice (Original)")
        ax1.axis('off')

        ax2 = plt.subplot(gs[0, 2:4])
        ax2.imshow(axial_slice, cmap='gray')
        ax2.imshow(seg_axial, alpha=0.5, cmap='viridis')
        ax2.set_title("Axial Slice with Segmentation")
        ax2.axis('off')

        # Coronal view (front)
        ax3 = plt.subplot(gs[1, 0:2])
        ax3.imshow(coronal_slice, cmap='gray')
        ax3.set_title("Coronal Slice (Original)")
        ax3.axis('off')

        ax4 = plt.subplot(gs[1, 2:4])
        ax4.imshow(coronal_slice, cmap='gray')
        ax4.imshow(seg_coronal, alpha=0.5, cmap='viridis')
        ax4.set_title("Coronal Slice with Segmentation")
        ax4.axis('off')

        # Sagittal view (side)
        ax5 = plt.subplot(gs[2, 0:2])
        ax5.imshow(sagittal_slice, cmap='gray')
        ax5.set_title("Sagittal Slice (Original)")
        ax5.axis('off')

        ax6 = plt.subplot(gs[2, 2:4])
        ax6.imshow(sagittal_slice, cmap='gray')
        ax6.imshow(seg_sagittal, alpha=0.5, cmap='viridis')
        ax6.set_title("Sagittal Slice with Segmentation")
        ax6.axis('off')

        # Add anomaly detection if provided
        if anomaly_score is not None:
            anomaly_axial = anomaly_score[:, :, z_mid]
            anomaly_coronal = anomaly_score[:, y_mid, :]
            anomaly_sagittal = anomaly_score[x_mid, :, :]

            ax7 = plt.subplot(gs[0, 4:6])
            im = ax7.imshow(anomaly_axial, cmap='hot', vmin=0, vmax=1)
            ax7.set_title("Axial Anomaly Score")
            ax7.axis('off')
            plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

            ax8 = plt.subplot(gs[1, 4:6])
            im = ax8.imshow(anomaly_coronal, cmap='hot', vmin=0, vmax=1)
            ax8.set_title("Coronal Anomaly Score")
            ax8.axis('off')
            plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)

            ax9 = plt.subplot(gs[2, 4:6])
            im = ax9.imshow(anomaly_sagittal, cmap='hot', vmin=0, vmax=1)
            ax9.set_title("Sagittal Anomaly Score")
            ax9.axis('off')
            plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)

        # Volumetric analysis
        ax_vol = plt.subplot(gs[3, 0:3])
        ax_vol.axis('off')

        # Calculate volumes
        unique_labels = np.unique(segmentation)
        volumes = {}
        total_brain_volume = 0

        # Create a mapping from segmentation class to names
        # Adjust based on your segmentation classes
        class_names = {
            0: "Background",
            1: "Gray Matter",
            2: "White Matter",
            3: "CSF",
            4: "Ventricles",
            5: "Hippocampus"
        }

        # Calculate the volume of a single voxel (assuming 1mm³ if not specified)
        voxel_volume = 1.0  # mm³

        # Calculate volumes
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Count voxels for this label
            voxel_count = np.sum(segmentation == label)
            volume_mm3 = voxel_count * voxel_volume
            volume_ml = volume_mm3 / 1000  # Convert to mL

            class_name = class_names.get(label, f"Class {label}")
            volumes[class_name] = volume_ml

            if label != 0:  # Skip background in total volume
                total_brain_volume += volume_ml

        # Create table data
        table_data = [["Structure", "Volume (mL)", "% of Total"]]
        for struct, vol in volumes.items():
            percentage = (vol / total_brain_volume) * 100 if total_brain_volume > 0 else 0
            table_data.append([struct, f"{vol:.2f}", f"{percentage:.2f}%"])

        # Add total row
        table_data.append(["Total Brain Volume", f"{total_brain_volume:.2f}", "100%"])

        # Create table
        table = ax_vol.table(
            cellText=table_data,
            colWidths=[0.5, 0.25, 0.25],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Highlight header
        for j in range(len(table_data[0])):
            cell = table[(0, j)]
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#CCCCFF')

        ax_vol.set_title("Volumetric Analysis", fontsize=14)

        # Clinical data visualization
        ax_clin = plt.subplot(gs[3, 3:6])
        ax_clin.axis('off')

        if clinical_data is not None:
            # Create a text box with clinical information
            clinical_text = "Clinical Data:\n\n"

            # Format key clinical metrics
            for key, value in clinical_data.items():
                if key not in ['PTID', 'subject_id']:  # Skip IDs
                    clinical_text += f"{key}: {value}\n"

            # Add clinical text
            ax_clin.text(0.05, 0.95, clinical_text,
                         transform=ax_clin.transAxes,
                         fontsize=11,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.5))
        else:
            ax_clin.text(0.5, 0.5, "No clinical data available",
                         transform=ax_clin.transAxes,
                         fontsize=12,
                         horizontalalignment='center',
                         verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.5))

        ax_clin.set_title("Clinical Information", fontsize=14)

        # Add footer with disclaimer
        fig.text(0.5, 0.01,
                 "This report is generated automatically and should be reviewed by a medical professional. " +
                 "Brain_AI v1.0.0",
                 ha='center', fontsize=10, style='italic')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save report if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, f"{subject_id}_report.png")
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            print(f"Report saved to: {report_path}")

        return fig

    def generate_cohort_report(results_df, clinical_df=None, output_dir=None):
        """
        Generate a report comparing metrics across different diagnostic groups.

        Args:
            results_df (pandas.DataFrame): DataFrame with volumetric results
            clinical_df (pandas.DataFrame, optional): DataFrame with clinical data
            output_dir (str, optional): Directory to save the report

        Returns:
            matplotlib.figure.Figure: Report figure
        """
        # Merge with clinical data if provided
        if clinical_df is not None:
            df = pd.merge(results_df, clinical_df, on='subject_id', how='left')
        else:
            df = results_df.copy()

        # Check if diagnostic group column exists
        if 'DX_GROUP' not in df.columns:
            print("Warning: No diagnostic group column found. Creating mock groups for demonstration.")
            # Create mock diagnostic groups for demonstration
            df['DX_GROUP'] = np.random.choice(['CN', 'MCI', 'AD'], size=len(df))

        # Create figure
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("Brain MRI Cohort Analysis Report", fontsize=20)

        # Define grid spec for layout
        gs = gridspec.GridSpec(3, 2)

        # Prepare data for visualization
        volume_cols = [col for col in df.columns if ('volume' in col.lower() or 'class' in col.lower())
                       and col != 'subject_id' and 'total' not in col.lower()]

        # Volume comparison by diagnostic group
        ax1 = plt.subplot(gs[0, 0])
        if volume_cols and len(df['DX_GROUP'].unique()) > 1:
            # Melt the DataFrame for easier plotting
            volume_df = df.melt(id_vars=['subject_id', 'DX_GROUP'],
                                value_vars=volume_cols,
                                var_name='Region',
                                value_name='Volume')

            # Create boxplot
            sns.boxplot(x='Region', y='Volume', hue='DX_GROUP', data=volume_df, ax=ax1)
            ax1.set_title("Brain Region Volumes by Diagnostic Group")
            ax1.set_xlabel("Brain Region")
            ax1.set_ylabel("Volume (mm³)")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(title="Diagnosis")
        else:
            ax1.text(0.5, 0.5, "Insufficient data for volume comparison",
                     ha='center', va='center', transform=ax1.transAxes)

        # Clinical scores comparison
        ax2 = plt.subplot(gs[0, 1])
        clinical_scores = ['MMSE', 'CDR', 'GDSCALE']
        available_scores = [col for col in clinical_scores if col in df.columns]

        if available_scores and 'DX_GROUP' in df.columns and len(df['DX_GROUP'].unique()) > 1:
            # Melt the DataFrame for clinical scores
            clin_df = df.melt(id_vars=['subject_id', 'DX_GROUP'],
                              value_vars=available_scores,
                              var_name='Clinical Score',
                              value_name='Score')

            # Create boxplot
            sns.boxplot(x='Clinical Score', y='Score', hue='DX_GROUP', data=clin_df, ax=ax2)
            ax2.set_title("Clinical Scores by Diagnostic Group")
            ax2.set_xlabel("Assessment")
            ax2.set_ylabel("Score")
            ax2.legend(title="Diagnosis")
        else:
            ax2.text(0.5, 0.5, "No clinical score data available",
                     ha='center', va='center', transform=ax2.transAxes)

        # Correlation heatmap between volumes and clinical scores
        ax3 = plt.subplot(gs[1, :])

        # Get numerical columns for correlation
        num_cols = volume_cols + available_scores

        if num_cols and len(num_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[num_cols].corr()

            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title("Correlation Between Brain Volumes and Clinical Scores")
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='y', rotation=0)
        else:
            ax3.text(0.5, 0.5, "Insufficient data for correlation analysis",
                     ha='center', va='center', transform=ax3.transAxes)

        # Group statistics table
        ax4 = plt.subplot(gs[2, :])
        ax4.axis('off')

        if 'DX_GROUP' in df.columns and volume_cols:
            # Group statistics
            group_stats = df.groupby('DX_GROUP')[volume_cols].agg(['mean', 'std', 'count'])

            # Flatten the MultiIndex
            group_stats.columns = [f"{col[0]}_{col[1]}" for col in group_stats.columns]

            # Reset index to convert to regular DataFrame
            group_stats = group_stats.reset_index()

            # Select columns to display
            display_cols = ['DX_GROUP'] + [col for col in group_stats.columns if 'mean' in col or 'count' in col]

            # Create table data
            table_data = []

            # Header row
            header = ['Diagnostic Group', 'Count']
            for col in [c for c in display_cols if 'mean' in c]:
                region = col.split('_')[0]
                header.append(f"{region} (mm³)")
            table_data.append(header)

            # Data rows
            for _, row in group_stats.iterrows():
                data_row = [row['DX_GROUP']]

                # Get count from any count column (they should all be the same)
                count_col = next(col for col in group_stats.columns if 'count' in col)
                data_row.append(str(int(row[count_col])))

                # Add mean volumes with standard deviation
                for col_mean in [c for c in display_cols if 'mean' in c]:
                    col_base = col_mean.replace('_mean', '')
                    col_std = f"{col_base}_std"

                    mean_val = row[col_mean]
                    std_val = row[col_std]

                    data_row.append(f"{mean_val:.2f} ± {std_val:.2f}")

                table_data.append(data_row)

            # Create table
            table = ax4.table(
                cellText=table_data,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Highlight header
            for j in range(len(table_data[0])):
                cell = table[(0, j)]
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#CCCCFF')

            ax4.set_title("Group Statistics (Mean ± Standard Deviation)", fontsize=14)
        else:
            ax4.text(0.5, 0.5, "Insufficient data for group statistics",
                     ha='center', va='center', transform=ax4.transAxes)

        # Add footer with date
        current_date = datetime.now().strftime("%Y-%m-%d")
        fig.text(0.5, 0.01,
                 f"Report generated on {current_date} | Brain_AI v1.0.0",
                 ha='center', fontsize=10, style='italic')

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save report if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "cohort_analysis_report.png")
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            print(f"Cohort report saved to: {report_path}")

        return fig
