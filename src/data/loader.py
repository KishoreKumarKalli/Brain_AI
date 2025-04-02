"""
Data loader module for brain segmentation framework.
Handles loading of MRI scans and clinical data.
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
import logging
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_loader')


class DataLoader:
    """
    Class for loading T1-weighted MRI scans and clinical data.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the DataLoader.

        Args:
            base_dir (str, optional): Base directory containing data folders
        """
        self.base_dir = base_dir
        self.mri_data_cache = {}  # Cache to avoid reloading the same files

        # Record initialization
        self.initialized_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.initialized_by = os.environ.get('USER', 'KishoreKumarKalli')

        logger.info(f"DataLoader initialized at {self.initialized_at} by {self.initialized_by}")
        if base_dir:
            logger.info(f"Base directory set to: {base_dir}")

    def load_nifti(self, file_path):
        """
        Load a NIfTI file and return the image data.

        Args:
            file_path (str): Path to the NIfTI file

        Returns:
            tuple: (image_data, image_obj) where image_data is a numpy array and
                  image_obj is the NiBabel image object
        """
        try:
            # Check if file is already cached
            if file_path in self.mri_data_cache:
                logger.info(f"Using cached data for: {file_path}")
                return self.mri_data_cache[file_path]

            logger.info(f"Loading NIfTI file: {file_path}")
            img = nib.load(file_path)
            data = img.get_fdata()

            # Cache the result
            self.mri_data_cache[file_path] = (data, img)

            return data, img
        except Exception as e:
            logger.error(f"Error loading NIfTI file {file_path}: {str(e)}")
            raise

    def load_mri_batch(self, directory_path, pattern="*.nii.gz", max_files=None):
        """
        Load a batch of MRI scans from a directory.

        Args:
            directory_path (str): Path to the directory containing MRI scans
            pattern (str): Glob pattern to match files
            max_files (int, optional): Maximum number of files to load

        Returns:
            dict: Dictionary mapping file paths to tuples of (image_data, image_obj)
        """
        try:
            logger.info(f"Loading batch of MRI scans from {directory_path} with pattern {pattern}")
            file_paths = glob.glob(os.path.join(directory_path, pattern))

            if max_files is not None:
                file_paths = file_paths[:max_files]
                logger.info(f"Limited to loading {max_files} files")

            result = {}
            for file_path in file_paths:
                try:
                    result[file_path] = self.load_nifti(file_path)
                except Exception as e:
                    logger.warning(f"Skipping {file_path} due to error: {str(e)}")
                    continue

            logger.info(f"Successfully loaded {len(result)} MRI scans")
            return result
        except Exception as e:
            logger.error(f"Error loading batch of MRI scans: {str(e)}")
            raise

    def load_clinical_data(self, file_path):
        """
        Load clinical data from a CSV file.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pandas.DataFrame: Loaded clinical data
        """
        try:
            logger.info(f"Loading clinical data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded clinical data with {data.shape[0]} rows and {data.shape[1]} columns")
            return data
        except Exception as e:
            logger.error(f"Error loading clinical data from {file_path}: {str(e)}")
            raise

    def load_all_clinical_data(self, directory_path):
        """
        Load all clinical CSV files from a directory.

        Args:
            directory_path (str): Path to the directory containing clinical CSV files

        Returns:
            dict: Dictionary mapping file names to DataFrames
        """
        try:
            logger.info(f"Loading all clinical data from {directory_path}")
            result = {}
            for file_name in os.listdir(directory_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(directory_path, file_name)
                    try:
                        result[file_name] = self.load_clinical_data(file_path)
                    except Exception as e:
                        logger.warning(f"Skipping {file_name} due to error: {str(e)}")
                        continue

            logger.info(f"Successfully loaded {len(result)} clinical data files")
            return result
        except Exception as e:
            logger.error(f"Error loading clinical data files: {str(e)}")
            raise

    def merge_clinical_and_mri_data(self, mri_data, clinical_df, id_column, id_mapping_func=None):
        """
        Merge MRI data with clinical data based on subject ID.

        Args:
            mri_data (dict): Dictionary of MRI data from load_mri_batch
            clinical_df (pandas.DataFrame): Clinical data
            id_column (str): Column name in clinical_df that contains subject IDs
            id_mapping_func (function, optional): Function to extract subject ID from MRI filename

        Returns:
            pandas.DataFrame: Merged data with MRI file paths and clinical data
        """
        try:
            logger.info("Merging MRI and clinical data")

            # Create a default ID mapping function if none provided
            if id_mapping_func is None:
                def default_mapping(filepath):
                    # Extract subject ID from filename (assumes filename contains subject ID)
                    filename = os.path.basename(filepath)
                    # Remove extension and get the first part (usually the subject ID)
                    subject_id = filename.split('_')[0]
                    return subject_id

                id_mapping_func = default_mapping

            # Create a DataFrame from MRI data
            mri_files = list(mri_data.keys())
            subject_ids = [id_mapping_func(filepath) for filepath in mri_files]

            mri_df = pd.DataFrame({
                'filepath': mri_files,
                'subject_id': subject_ids
            })

            # Merge with clinical data
            merged_df = pd.merge(
                mri_df,
                clinical_df,
                left_on='subject_id',
                right_on=id_column,
                how='inner'
            )

            logger.info(f"Merged data has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
            return merged_df
        except Exception as e:
            logger.error(f"Error merging MRI and clinical data: {str(e)}")
            raise

    def load_and_merge_adni_data(self, mri_base_dir, clinical_base_dir, max_files=None):
        """
        Load and merge all ADNI data (MRI and clinical).

        Args:
            mri_base_dir (str): Base directory containing MRI data folders (CN, MCI, AD)
            clinical_base_dir (str): Directory containing clinical CSV files
            max_files (int, optional): Maximum number of files to load per category

        Returns:
            tuple: (merged_data, clinical_data_dict, mri_data_dict)
        """
        try:
            # Current date and time for logging
            current_time = "2025-04-02 13:48:39"
            logger.info(f"Starting ADNI data loading and merging process at {current_time}")
            logger.info(f"Process executed by: KishoreKumarKalli")

            # Load clinical data
            clinical_data_dict = self.load_all_clinical_data(clinical_base_dir)

            # Load main subject information
            adni_t1_df = clinical_data_dict.get('ADNI_T1.csv')
            if adni_t1_df is None:
                raise ValueError("ADNI_T1.csv not found in clinical data directory")

            # Load MRI data for each diagnostic group
            mri_data_dict = {}

            for group in ['CN', 'MCI', 'AD']:
                group_dir = os.path.join(mri_base_dir, group)
                if os.path.exists(group_dir):
                    logger.info(f"Loading {group} MRI data")
                    mri_data_dict[group] = self.load_mri_batch(group_dir, max_files=max_files)

            # Merge data for each group
            merged_data = pd.DataFrame()

            for group, mri_data in mri_data_dict.items():
                if mri_data:
                    # Create a group-specific merge
                    group_df = self.merge_clinical_and_mri_data(
                        mri_data,
                        adni_t1_df,
                        'Subject_ID'
                    )

                    # Add diagnostic group label
                    group_df['DiagnosticGroup'] = group

                    # Append to the merged data
                    merged_data = pd.concat([merged_data, group_df], ignore_index=True)

            logger.info(f"Final merged data has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns")
            return merged_data, clinical_data_dict, mri_data_dict

        except Exception as e:
            logger.error(f"Error in load_and_merge_adni_data: {str(e)}")
            raise


# Helper functions for extracting metadata from file paths
def extract_subject_id_from_filename(filepath):
    """
    Extract subject ID from a NIfTI filename.
    Example: 'sub-ADNI123_T1w.nii.gz' -> 'ADNI123'

    Args:
        filepath (str): Path to the NIfTI file

    Returns:
        str: Subject ID
    """
    filename = os.path.basename(filepath)

    # Handle different naming conventions
    if 'sub-' in filename:
        # BIDS format: sub-<subject_id>_<modality>.nii.gz
        subject_part = filename.split('_')[0]
        subject_id = subject_part.replace('sub-', '')
    else:
        # Other formats
        subject_id = filename.split('_')[0]

    return subject_id


def get_modality_from_filename(filepath):
    """
    Extract modality information from a NIfTI filename.
    Example: 'sub-ADNI123_T1w.nii.gz' -> 'T1w'

    Args:
        filepath (str): Path to the NIfTI file

    Returns:
        str: Modality
    """
    filename = os.path.basename(filepath)

    # BIDS format: sub-<subject_id>_<modality>.nii.gz
    parts = filename.split('_')

    if len(parts) > 1:
        # Look for parts that might indicate modality
        for part in parts[1:]:
            if any(mod in part.lower() for mod in ['t1', 't2', 'flair', 'dwi', 'bold']):
                return part.split('.')[0]  # Remove file extension

    # Default if modality not found
    return "unknown"