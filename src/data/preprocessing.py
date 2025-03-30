import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
import yaml


class DataPreprocessor:
    """Class for preprocessing MRI data"""

    def __init__(self, config_path='config.yml'):
        """Initialize the preprocessor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.raw_dir = self.config['data']['raw_dir']
        self.processed_dir = self.config['data']['processed_dir']
        self.clinical_dir = self.config['data']['clinical_dir']
        self.target_shape = self.config['preprocessing']['target_shape']

    def load_clinical_data(self):
        """Load and merge all clinical data from CSV files."""
        # Load main subject data
        subjects_df = pd.read_csv(os.path.join(self.clinical_dir, 'ADNI_T1.csv'))

        # Load additional clinical data
        cdr_df = pd.read_csv(os.path.join(self.clinical_dir, 'CDR.csv'))
        mmse_df = pd.read_csv(os.path.join(self.clinical_dir, 'MMSE.csv'))
        gds_df = pd.read_csv(os.path.join(self.clinical_dir, 'GDSCALE.csv'))
        adas1_df = pd.read_csv(os.path.join(self.clinical_dir, 'ADAS_ADNI1.csv'))
        neurobat_df = pd.read_csv(os.path.join(self.clinical_dir, 'NEUROBAT.csv'))
        ptdemog_df = pd.read_csv(os.path.join(self.clinical_dir, 'PTDEMOG.csv'))
        adas23_df = pd.read_csv(os.path.join(self.clinical_dir, 'ADAS_ADNIGO23.csv'))

        # Merge dataframes (assuming common subject IDs)
        # Note: Adjust column names as needed based on actual data
        merged_df = subjects_df

        for df, name in zip([cdr_df, mmse_df, gds_df, adas1_df, neurobat_df, ptdemog_df, adas23_df],
                            ['CDR', 'MMSE', 'GDS', 'ADAS1', 'NEUROBAT', 'PTDEMOG', 'ADAS23']):
            merged_df = pd.merge(merged_df, df, on='SUBJECT_ID', how='left', suffixes=('', f'_{name}'))

        return merged_df

    def load_nifti(self, file_path):
        """Load a NIfTI file and return its data array and affine matrix."""
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata(), nifti_img.affine

    def normalize_scan(self, scan_data, method='min_max'):
        """Normalize scan intensity."""
        if method == 'min_max':
            min_val = np.min(scan_data)
            max_val = np.max(scan_data)

            if max_val == min_val:
                return np.zeros_like(scan_data)

            normalized = (scan_data - min_val) / (max_val - min_val)
            return normalized

        elif method == 'z_score':
            mean_val = np.mean(scan_data)
            std_val = np.std(scan_data)

            if std_val == 0:
                return np.zeros_like(scan_data)

            normalized = (scan_data - mean_val) / std_val
            return normalized

        else:
            raise ValueError(f"Normalization method '{method}' not supported")

    def resize_scan(self, scan_data, target_shape=None):
        """Resize scan to target dimensions."""
        if target_shape is None:
            target_shape = self.target_shape

        # Calculate zoom factors
        factors = [t / s for t, s in zip(target_shape, scan_data.shape)]

        # Apply zoom
        resized = zoom(scan_data, factors, order=1)

        return resized

    def preprocess_scan(self, scan_path, output_path=None):
        """Preprocess a single scan: load, normalize, resize."""
        # Load scan
        scan_data, affine = self.load_nifti(scan_path)

        # Normalize intensity
        normalized = self.normalize_scan(
            scan_data,
            method=self.config['preprocessing']['intensity_normalization']
        )

        # Resize to standard dimensions
        resized = self.resize_scan(normalized)

        if output_path:
            # Save preprocessed scan
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nib.Nifti1Image(resized, affine), output_path)

        return resized, affine

    def preprocess_all_data(self):
        """Preprocess all scans in the dataset."""
        # Create output directories
        os.makedirs(self.processed_dir, exist_ok=True)

        # Process each diagnostic group
        for group in ['CN', 'MCI', 'AD']:
            input_dir = os.path.join(self.raw_dir, group)
            output_dir = os.path.join(self.processed_dir, group)
            os.makedirs(output_dir, exist_ok=True)

            # Get all NIfTI files in the directory
            nifti_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

            for file_name in nifti_files:
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)
                self.preprocess_scan(input_path, output_path)
                print(f"Processed: {file_name} -> {output_path}")

    def create_data_splits(self):
        """Create train/validation/test splits from the dataset."""
        # Load subject data
        subjects_df = pd.read_csv(os.path.join(self.clinical_dir, 'ADNI_T1.csv'))

        # Create splits
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        test_ratio = self.config['data']['test_ratio']

        # Verify ratios sum to 1
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-10, "Split ratios must sum to 1"

        # Split by diagnostic group to maintain class balance
        splits = {'train': [], 'val': [], 'test': []}

        for group in subjects_df['DX_GROUP'].unique():
            group_df = subjects_df[subjects_df['DX_GROUP'] == group]

            # Calculate split sizes
            n_samples = len(group_df)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            # Create random indices for splitting
            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            # Add to splits
            splits['train'].append(group_df.iloc[train_indices])
            splits['val'].append(group_df.iloc[val_indices])
            splits['test'].append(group_df.iloc[test_indices])

        # Concatenate all groups
        for split in splits:
            splits[split] = pd.concat(splits[split])

            # Save to CSV
            output_path = os.path.join(self.processed_dir, f'{split}_subjects.csv')
            splits[split].to_csv(output_path, index=False)
            print(f"Created {split} split: {len(splits[split])} subjects -> {output_path}")

        return splits


# Example usage in main.py
if __name__ == "__main__":
    preprocessor = DataPreprocessor('config.yml')
    preprocessor.preprocess_all_data()
    splits = preprocessor.create_data_splits()
    clinical_data = preprocessor.load_clinical_data()
    clinical_data.to_csv('data/processed/merged_clinical_data.csv', index=False)