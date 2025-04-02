"""
Volumetric analysis module for brain segmentation framework.
This module provides functionality for volumetric analysis of brain structures
from segmentation masks.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import logging
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('volumetric_analysis')


class VolumetricAnalyzer:
    """
    Class for volumetric analysis of brain structures from segmentation masks.
    """

    def __init__(self, output_dir="./volumetric_results"):
        """
        Initialize the volumetric analyzer.

        Args:
            output_dir (str): Directory to save analysis results
        """
        # Metadata
        self.analysis_date = "2025-04-02 14:54:39"
        self.analyst = "KishoreKumarKalli"

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Structure label mapping (customize as needed)
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

        # Initialize results storage
        self.results = {}

        logger.info(f"Volumetric analyzer initialized at {self.analysis_date} by {self.analyst}")
        logger.info(f"Results will be saved to: {output_dir}")

    def load_segmentation(self, filepath):
        """
        Load a segmentation mask from a NIfTI file.

        Args:
            filepath (str): Path to the segmentation mask file

        Returns:
            tuple: (data, nib.Nifti1Image) - segmentation data array and NiBabel image object
        """
        try:
            logger.info(f"Loading segmentation mask from {filepath}")
            img = nib.load(filepath)
            data = img.get_fdata()

            # Check data type and convert if necessary
            if not np.issubdtype(data.dtype, np.integer):
                logger.warning(f"Segmentation data is not integer type (found {data.dtype}). Converting to int16.")
                data = np.round(data).astype(np.int16)

            # Get unique labels
            unique_labels = np.unique(data)
            logger.info(f"Segmentation contains labels: {unique_labels}")

            return data, img
        except Exception as e:
            logger.error(f"Error loading segmentation mask: {str(e)}")
            raise

    def calculate_volumes(self, seg_data, img, output_path=None):
        """
        Calculate volumes of brain structures from segmentation mask.

        Args:
            seg_data (numpy.ndarray): Segmentation data
            img (nib.Nifti1Image): NiBabel image object with affine transformation
            output_path (str, optional): Path to save the results CSV

        Returns:
            pandas.DataFrame: Table of structure volumes
        """
        logger.info("Calculating volumes for brain structures")

        # Get voxel dimensions from image affine
        voxel_dims = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
        voxel_volume = np.prod(voxel_dims)

        logger.info(f"Voxel dimensions: {voxel_dims} mm")
        logger.info(f"Voxel volume: {voxel_volume} mm³")

        # Get unique labels
        labels = np.unique(seg_data)
        labels = labels[labels != 0]  # Exclude background

        # Calculate volumes
        volumes = []
        for label in labels:
            # Count voxels with this label
            voxel_count = np.sum(seg_data == label)

            # Calculate volume in mm³
            volume_mm3 = voxel_count * voxel_volume

            # Calculate volume in ml (1 cm³ = 1 ml)
            volume_ml = volume_mm3 / 1000

            # Get structure name
            structure_name = self.structure_labels.get(label, f"Unknown-{label}")

            volumes.append({
                'label': int(label),
                'structure': structure_name,
                'voxel_count': voxel_count,
                'volume_mm3': volume_mm3,
                'volume_ml': volume_ml
            })

        # Create DataFrame
        volumes_df = pd.DataFrame(volumes)

        # Calculate total brain volume (excluding background and CSF)
        brain_labels = [label for label in labels if label not in [0, 3, 5]]  # Exclude background, CSF, ventricles
        total_brain_volume = sum(volumes_df[volumes_df['label'].isin(brain_labels)]['volume_ml'])

        # Calculate ICV (Intracranial Volume - all tissues including CSF)
        icv = sum(volumes_df['volume_ml'])

        # Add total volumes to results
        results = {
            'structure_volumes': volumes_df,
            'total_brain_volume': total_brain_volume,
            'intracranial_volume': icv
        }

        logger.info(f"Total brain volume: {total_brain_volume:.2f} ml")
        logger.info(f"Intracranial volume: {icv:.2f} ml")

        # Save to CSV if output path provided
        if output_path:
            volumes_df.to_csv(output_path, index=False)
            logger.info(f"Saved volume results to {output_path}")

        # Store in results dictionary
        self.results['volumes'] = results

        return results

    def calculate_relative_volumes(self, volumes_result, output_path=None):
        """
        Calculate relative volumes of brain structures as percentage of ICV.

        Args:
            volumes_result (dict): Result from calculate_volumes method
            output_path (str, optional): Path to save the results CSV

        Returns:
            pandas.DataFrame: Table of relative structure volumes
        """
        logger.info("Calculating relative volumes as percentage of ICV")

        volumes_df = volumes_result['structure_volumes']
        icv = volumes_result['intracranial_volume']

        # Calculate relative volumes
        rel_volumes_df = volumes_df.copy()
        rel_volumes_df['volume_percentage_of_icv'] = (volumes_df['volume_ml'] / icv) * 100

        # Calculate relative volumes as percentage of total brain volume
        brain_volume = volumes_result['total_brain_volume']
        rel_volumes_df['volume_percentage_of_brain'] = (volumes_df['volume_ml'] / brain_volume) * 100

        # Save to CSV if output path provided
        if output_path:
            rel_volumes_df.to_csv(output_path, index=False)
            logger.info(f"Saved relative volume results to {output_path}")

        # Store in results dictionary
        self.results['relative_volumes'] = rel_volumes_df

        return rel_volumes_df

    def extract_surface_mesh(self, seg_data, label, smooth=True, output_file=None):
        """
        Extract surface mesh for a specific structure from segmentation.

        Args:
            seg_data (numpy.ndarray): Segmentation data
            label (int): Label of the structure to extract
            smooth (bool): Whether to smooth the mesh
            output_file (str, optional): Path to save the mesh

        Returns:
            dict: Mesh data (vertices, faces)
        """
        logger.info(f"Extracting surface mesh for structure with label {label}")

        # Create binary mask for the specified label
        binary = seg_data == label

        # Extract surface mesh using marching cubes algorithm
        try:
            vertices, faces, normals, values = measure.marching_cubes(binary, level=0.5, spacing=(1.0, 1.0, 1.0))

            # Smooth mesh if requested
            if smooth:
                try:
                    from scipy.ndimage.filters import gaussian_filter
                    for _ in range(10):
                        # Apply Gaussian smoothing to vertices
                        for i in range(3):
                            vertices[:, i] = gaussian_filter(vertices[:, i], sigma=1)
                    logger.info("Mesh smoothing applied")
                except Exception as e:
                    logger.warning(f"Mesh smoothing failed: {str(e)}")

            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'normals': normals,
                'values': values
            }

            # Save mesh if output file provided
            if output_file:
                try:
                    # Save as OBJ format
                    with open(output_file, 'w') as f:
                        # Write vertices
                        for v in vertices:
                            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                        # Write faces (OBJ uses 1-indexed vertices)
                        for face in faces:
                            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                    logger.info(f"Saved mesh to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving mesh: {str(e)}")

            return mesh_data

        except Exception as e:
            logger.error(f"Error extracting surface mesh: {str(e)}")
            return None

    def visualize_segmentation(self, seg_data, output_file=None, slice_idx=None, alpha=0.7):
        """
        Visualize segmentation mask using slice views.

        Args:
            seg_data (numpy.ndarray): Segmentation data
            output_file (str, optional): Path to save the visualization
            slice_idx (tuple, optional): Slice indices (x, y, z)
            alpha (float): Transparency for overlay

        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Create a colormap for the segmentation labels
        import matplotlib.colors as mcolors
        n_labels = max(max(np.unique(seg_data)), 7) + 1  # Ensure at least 8 colors (0-7)
        colors = plt.cm.viridis(np.linspace(0, 1, n_labels))
        colors[0] = [0, 0, 0, 0]  # Make background transparent
        cmap = mcolors.ListedColormap(colors)

        # Get dimensions
        x_dim, y_dim, z_dim = seg_data.shape

        # Select middle slices if not specified
        if slice_idx is None:
            slice_idx = (x_dim // 2, y_dim // 2, z_dim // 2)

        # Create figure with three slice views
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Sagittal view (YZ plane)
        axes[0].imshow(seg_data[slice_idx[0], :, :].T, cmap=cmap, interpolation='nearest', origin='lower')
        axes[0].set_title(f"Sagittal View (x={slice_idx[0]})")
        axes[0].set_xlabel('Y axis')
        axes[0].set_ylabel('Z axis')

        # Coronal view (XZ plane)
        axes[1].imshow(seg_data[:, slice_idx[1], :].T, cmap=cmap, interpolation='nearest', origin='lower')
        axes[1].set_title(f"Coronal View (y={slice_idx[1]})")
        axes[1].set_xlabel('X axis')
        axes[1].set_ylabel('Z axis')

        # Axial view (XY plane)
        axes[2].imshow(seg_data[:, :, slice_idx[2]], cmap=cmap, interpolation='nearest', origin='lower')
        axes[2].set_title(f"Axial View (z={slice_idx[2]})")
        axes[2].set_xlabel('X axis')
        axes[2].set_ylabel('Y axis')

        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=self.structure_labels.get(i, f"Unknown-{i}"))
                           for i in range(1, n_labels) if i in np.unique(seg_data)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(legend_elements))

        # Add title with metadata
        plt.suptitle(f"Brain Segmentation Visualization\nGenerated: 2025-04-02 14:55:40 by KishoreKumarKalli",
                     fontsize=14)

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Save figure if output file provided
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved segmentation visualization to {output_file}")

        return fig

    def calculate_shape_metrics(self, seg_data, label, output_path=None):
        """
        Calculate shape metrics for a specific structure.

        Args:
            seg_data (numpy.ndarray): Segmentation data
            label (int): Label of the structure to analyze
            output_path (str, optional): Path to save the results

        Returns:
            dict: Shape metrics
        """
        logger.info(f"Calculating shape metrics for structure with label {label}")

        # Create binary mask for the specified label
        binary = seg_data == label

        structure_name = self.structure_labels.get(label, f"Unknown-{label}")

        # Calculate basic metrics
        volume_voxels = np.sum(binary)

        if volume_voxels == 0:
            logger.warning(f"No voxels found for label {label} ({structure_name})")
            return None

        # Calculate surface area using marching cubes
        try:
            verts, faces, _, _ = measure.marching_cubes(binary, level=0.5, spacing=(1.0, 1.0, 1.0))
            surface_area = measure.mesh_surface_area(verts, faces)
        except Exception as e:
            logger.warning(f"Error calculating surface area: {str(e)}")
            surface_area = None

        # Calculate shape descriptors
        try:
            # Get properties of labeled regions
            props = measure.regionprops(binary.astype(int))[0]

            # Sphericity (1 for a perfect sphere)
            if surface_area:
                sphericity = (36 * np.pi * volume_voxels ** 2) ** (1 / 3) / surface_area
            else:
                sphericity = None

            # Other shape metrics
            metrics = {
                'label': label,
                'structure': structure_name,
                'volume_voxels': volume_voxels,
                'surface_area': surface_area,
                'sphericity': sphericity,
                'equivalent_diameter': props.equivalent_diameter,
                'major_axis_length': props.major_axis_length,
                'minor_axis_length': props.minor_axis_length,
                'eccentricity': props.eccentricity,
                'solidity': props.solidity,
                'extent': props.extent,
            }
        except Exception as e:
            logger.error(f"Error calculating shape metrics: {str(e)}")
            return None

        # Save results if output path provided
        if output_path:
            pd.DataFrame([metrics]).to_csv(output_path, index=False)
            logger.info(f"Saved shape metrics to {output_path}")

        # Store in results dictionary
        if 'shape_metrics' not in self.results:
            self.results['shape_metrics'] = {}
        self.results['shape_metrics'][label] = metrics

        return metrics

    def batch_process_segmentations(self, segmentation_files, subject_ids=None):
        """
        Process multiple segmentation files and compile results.

        Args:
            segmentation_files (list): List of paths to segmentation files
            subject_ids (list, optional): List of subject IDs corresponding to files

        Returns:
            pandas.DataFrame: Combined results for all subjects
        """
        logger.info(f"Batch processing {len(segmentation_files)} segmentation files")

        # Create results storage
        all_volumes = []
        all_shape_metrics = []

        # Process each segmentation file
        for i, seg_file in enumerate(segmentation_files):
            try:
                # Get subject ID
                if subject_ids and i < len(subject_ids):
                    subject_id = subject_ids[i]
                else:
                    # Extract subject ID from filename
                    subject_id = os.path.basename(seg_file).split('_')[0]

                logger.info(f"Processing subject {subject_id} ({i + 1}/{len(segmentation_files)})")

                # Load segmentation
                seg_data, img = self.load_segmentation(seg_file)

                # Calculate volumes
                volumes = self.calculate_volumes(seg_data, img)

                # Add subject ID to volumes
                volumes_df = volumes['structure_volumes'].copy()
                volumes_df['subject_id'] = subject_id

                # Add ICV and TBV
                volumes_df['total_brain_volume_ml'] = volumes['total_brain_volume']
                volumes_df['intracranial_volume_ml'] = volumes['intracranial_volume']

                # Add to all volumes
                all_volumes.append(volumes_df)

                # Calculate shape metrics for each structure
                labels = np.unique(seg_data)
                labels = labels[labels != 0]  # Exclude background

                for label in labels:
                    metrics = self.calculate_shape_metrics(seg_data, label)
                    if metrics:
                        metrics_df = pd.DataFrame([metrics])
                        metrics_df['subject_id'] = subject_id
                        all_shape_metrics.append(metrics_df)

                logger.info(f"Completed processing for subject {subject_id}")

            except Exception as e:
                logger.error(f"Error processing file {seg_file}: {str(e)}")
                continue

        # Combine all results
        if all_volumes:
            combined_volumes = pd.concat(all_volumes, ignore_index=True)
            volumes_output = os.path.join(self.output_dir, 'combined_volumes.csv')
            combined_volumes.to_csv(volumes_output, index=False)
            logger.info(f"Saved combined volumes to {volumes_output}")
        else:
            combined_volumes = pd.DataFrame()
            logger.warning("No volume data to combine")

        if all_shape_metrics:
            combined_metrics = pd.concat(all_shape_metrics, ignore_index=True)
            metrics_output = os.path.join(self.output_dir, 'combined_shape_metrics.csv')
            combined_metrics.to_csv(metrics_output, index=False)
            logger.info(f"Saved combined shape metrics to {metrics_output}")
        else:
            combined_metrics = pd.DataFrame()
            logger.warning("No shape metrics data to combine")

        # Create summary statistics
        if not combined_volumes.empty:
            # Summary by structure
            structure_summary = combined_volumes.groupby('structure').agg({
                'volume_ml': ['mean', 'std', 'min', 'max'],
                'volume_percentage_of_icv': ['mean',
                                             'std'] if 'volume_percentage_of_icv' in combined_volumes.columns else None
            }).reset_index()

            summary_output = os.path.join(self.output_dir, 'volume_summary_by_structure.csv')
            structure_summary.to_csv(summary_output)
            logger.info(f"Saved volume summary to {summary_output}")

            # Create visualization of volume distributions
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='structure', y='volume_ml', data=combined_volumes)
            plt.title(f'Volume Distribution by Brain Structure\nGenerated: 2025-04-02 14:55:40')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            boxplot_output = os.path.join(self.output_dir, 'volume_distribution_boxplot.png')
            plt.savefig(boxplot_output, dpi=300)
            plt.close()
            logger.info(f"Saved volume distribution boxplot to {boxplot_output}")

        logger.info("Batch processing completed")

        return {
            'volumes': combined_volumes,
            'shape_metrics': combined_metrics
        }


# Utility functions

def calculate_asymmetry_index(left_volume, right_volume):
    """
    Calculate asymmetry index between left and right structures.

    Args:
        left_volume (float): Volume of left structure
        right_volume (float): Volume of right structure

    Returns:
        float: Asymmetry index (ranges from -2 to 2, 0 indicates perfect symmetry)
    """
    if left_volume == 0 and right_volume == 0:
        return 0

    asymmetry = (left_volume - right_volume) / ((left_volume + right_volume) / 2)
    return asymmetry


def voxel_count_to_volume(voxel_count, voxel_dims):
    """
    Convert voxel count to physical volume.

    Args:
        voxel_count (int): Number of voxels
        voxel_dims (tuple): Voxel dimensions in mm (x, y, z)

    Returns:
        tuple: (volume_mm3, volume_ml)
    """
    voxel_volume = np.prod(voxel_dims)
    volume_mm3 = voxel_count * voxel_volume
    volume_ml = volume_mm3 / 1000  # 1 ml = 1000 mm³

    return volume_mm3, volume_ml