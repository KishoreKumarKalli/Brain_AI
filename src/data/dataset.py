import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
    RandAffined,
    SpatialPadd,
)


class BrainMRIDataset(Dataset):
    """
    Dataset class for brain MRI data with clinical information.

    Args:
        root_dir (str): Root directory containing the data folder structure
        split (str): One of 'train', 'val', or 'test'
        transform (callable, optional): Optional transform to be applied on a sample
        diagnosis_groups (list, optional): List of diagnosis groups to include (e.g., ["CN", "MCI", "AD"])
    """

    def __init__(self, root_dir, split='train', transform=None, diagnosis_groups=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Default to all diagnosis groups if not specified
        if diagnosis_groups is None:
            diagnosis_groups = ["CN", "MCI", "AD"]
        self.diagnosis_groups = diagnosis_groups

        # Load primary clinical data
        adni_t1_path = self.root_dir / "data" / "clinical" / "ADNI_T1.csv"
        self.adni_t1_data = pd.read_csv(adni_t1_path)

        # Create mapping of diagnosis to numeric labels
        self.diagnosis_to_label = {'CN': 0, 'MCI': 1, 'AD': 2}

        # Load additional clinical data for enriching the dataset
        self.clinical_data = self._load_clinical_data()

        # Create list of samples
        self.samples = self._create_samples()

    def _load_clinical_data(self):
        """Load and merge relevant clinical data."""
        clinical_data = {}
        clinical_files = {
            "CDR": self.root_dir / "data" / "clinical" / "CDR.csv",
            "MMSE": self.root_dir / "data" / "clinical" / "MMSE.csv",
            "GDSCALE": self.root_dir / "data" / "clinical" / "GDSCALE.csv",
            "ADAS_ADNI1": self.root_dir / "data" / "clinical" / "ADAS_ADNI1.csv",
            "NEUROBAT": self.root_dir / "data" / "clinical" / "NEUROBAT.csv",
            "PTDEMOG": self.root_dir / "data" / "clinical" / "PTDEMOG.csv",
            "ADAS_ADNIGO23": self.root_dir / "data" / "clinical" / "ADAS_ADNIGO23.csv"
        }

        for name, path in clinical_files.items():
            if path.exists():
                clinical_data[name] = pd.read_csv(path)

        return clinical_data

    def _create_samples(self):
        """Create a list of samples based on the available data and split."""
        samples = []

        # Filter the subjects based on diagnosis groups
        filtered_data = self.adni_t1_data[self.adni_t1_data['DX_GROUP'].isin(self.diagnosis_groups)]

        # Implement train/val/test split (80/10/10 by default)
        # In real project, you would use a proper stratified split
        # This is a simple implementation for demonstration
        unique_subjects = filtered_data['PTID'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_subjects)

        n_subjects = len(unique_subjects)
        if self.split == 'train':
            selected_subjects = unique_subjects[:int(0.8 * n_subjects)]
        elif self.split == 'val':
            selected_subjects = unique_subjects[int(0.8 * n_subjects):int(0.9 * n_subjects)]
        else:  # test
            selected_subjects = unique_subjects[int(0.9 * n_subjects):]

        # Filter data to only include selected subjects
        split_data = filtered_data[filtered_data['PTID'].isin(selected_subjects)]

        # Create sample entries
        for _, row in split_data.iterrows():
            subject_id = row['PTID']
            diagnosis = row['DX_GROUP']

            # Construct the path to the MRI file based on diagnosis
            mri_path = self.root_dir / "data" / "raw" / diagnosis / f"{subject_id}.nii"

            if not mri_path.exists():
                # Try common alternative file extensions
                for ext in ['.nii.gz', '.nii']:
                    alt_path = self.root_dir / "data" / "raw" / diagnosis / f"{subject_id}{ext}"
                    if alt_path.exists():
                        mri_path = alt_path
                        break

            # Only include sample if MRI file exists
            if mri_path.exists():
                # Gather additional clinical data for this subject
                clinical_metrics = self._get_clinical_metrics(subject_id)

                sample = {
                    'image_path': str(mri_path),
                    'subject_id': subject_id,
                    'diagnosis': diagnosis,
                    'label': self.diagnosis_to_label[diagnosis],
                    'clinical_data': clinical_metrics
                }
                samples.append(sample)

        return samples

    def _get_clinical_metrics(self, subject_id):
        """Extract relevant clinical metrics for a given subject."""
        metrics = {}

        # Extract CDR score if available
        if 'CDR' in self.clinical_data:
            cdr_data = self.clinical_data['CDR']
            subject_cdr = cdr_data[cdr_data['PTID'] == subject_id]
            if not subject_cdr.empty:
                metrics['CDR_GLOBAL'] = subject_cdr['CDGLOBAL'].values[0]

        # Extract MMSE score if available
        if 'MMSE' in self.clinical_data:
            mmse_data = self.clinical_data['MMSE']
            subject_mmse = mmse_data[mmse_data['PTID'] == subject_id]
            if not subject_mmse.empty:
                metrics['MMSE_SCORE'] = subject_mmse['MMSETOTAL'].values[0]

        # Add more clinical metrics as needed
        # Example: Add GDSCALE score
        if 'GDSCALE' in self.clinical_data:
            gds_data = self.clinical_data['GDSCALE']
            subject_gds = gds_data[gds_data['PTID'] == subject_id]
            if not subject_gds.empty:
                metrics['GDS_SCORE'] = subject_gds['GDTOTAL'].values[0]

        # Demographics
        if 'PTDEMOG' in self.clinical_data:
            demog_data = self.clinical_data['PTDEMOG']
            subject_demog = demog_data[demog_data['PTID'] == subject_id]
            if not subject_demog.empty:
                metrics['AGE'] = subject_demog['AGE'].values[0]
                metrics['GENDER'] = subject_demog['PTGENDER'].values[0]
                metrics['EDUCATION'] = subject_demog['PTEDUCAT'].values[0]

        return metrics

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load the MRI data
        mri_img = nib.load(sample['image_path'])
        mri_data = mri_img.get_fdata()

        # Convert to tensor and ensure channel first (C, D, H, W)
        mri_tensor = torch.from_numpy(mri_data).float()
        if len(mri_tensor.shape) == 3:  # If no channel dimension
            mri_tensor = mri_tensor.unsqueeze(0)

        # Apply transforms if specified
        if self.transform:
            mri_tensor = self.transform(mri_tensor)

        return {
            'image': mri_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'subject_id': sample['subject_id'],
            'diagnosis': sample['diagnosis'],
            'clinical_data': sample['clinical_data']
        }


def get_transforms(mode='train'):
    """
    Get transforms for data preprocessing and augmentation.

    Args:
        mode (str): One of 'train', 'val', or 'test'

    Returns:
        monai.transforms.Compose: Composed transforms
    """
    if mode == 'train':
        return Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Orientationd(keys=['image'], axcodes="RAS"),
            Spacingd(keys=['image'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear')),
            CropForegroundd(keys=['image'], source_key='image'),
            SpatialPadd(keys=['image'], spatial_size=(128, 128, 128)),
            RandFlipd(keys=['image'], prob=0.5),
            RandRotate90d(keys=['image'], prob=0.5, spatial_axes=(0, 1)),
            RandShiftIntensityd(keys=['image'], prob=0.5, offsets=0.1),
            RandAffined(
                keys=['image'],
                prob=0.5,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                scale_range=(0.05, 0.05, 0.05),
                mode=('bilinear'),
                padding_mode='zeros'
            ),
            ToTensord(keys=['image'])
        ])
    else:  # val or test
        return Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            ScaleIntensityd(keys=['image']),
            Orientationd(keys=['image'], axcodes="RAS"),
            Spacingd(keys=['image'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear')),
            CropForegroundd(keys=['image'], source_key='image'),
            SpatialPadd(keys=['image'], spatial_size=(128, 128, 128)),
            ToTensord(keys=['image'])
        ])


def create_data_loaders(root_dir, batch_size=4, num_workers=4):
    """
    Create data loaders for training, validation, and testing.

    Args:
        root_dir (str): Root directory containing the data folder structure
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading

    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    # Define transforms
    train_transforms = get_transforms(mode='train')
    val_transforms = get_transforms(mode='val')
    test_transforms = get_transforms(mode='test')

    # Create datasets
    train_ds = BrainMRIDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transforms,
        diagnosis_groups=["CN", "MCI", "AD"]
    )

    val_ds = BrainMRIDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transforms,
        diagnosis_groups=["CN", "MCI", "AD"]
    )

    test_ds = BrainMRIDataset(
        root_dir=root_dir,
        split='test',
        transform=test_transforms,
        diagnosis_groups=["CN", "MCI", "AD"]
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }