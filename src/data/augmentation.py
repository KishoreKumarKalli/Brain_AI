import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    RandBiasFieldd,
    RandGibbsNoised,
    OneOf,
    ToTensord,
    EnsureTyped,
)
from monai.config import KeysCollection
from typing import Dict, Hashable, Mapping, Optional, Union


class AugmentationFactory:
    """
    Factory class to create data augmentation pipelines for brain MRI data.
    Provides configurable augmentation strategies for different types of
    segmentation and classification tasks.
    """

    @staticmethod
    def get_classification_transforms(keys: KeysCollection, prob: float = 0.5,
                                      addl_keys: Optional[KeysCollection] = None):
        """
        Get transforms for classification tasks.

        Args:
            keys (KeysCollection): The keys to apply transforms on (typically 'image')
            prob (float): Probability of applying each transform
            addl_keys (KeysCollection, optional): Additional keys to transform (e.g., 'label')

        Returns:
            Compose: Composed transforms
        """
        all_keys = [keys] if isinstance(keys, (str, int)) else list(keys)
        if addl_keys is not None:
            if isinstance(addl_keys, (str, int)):
                all_keys.append(addl_keys)
            else:
                all_keys.extend(list(addl_keys))

        return Compose([
            # Spatial transformations
            RandFlipd(keys=keys, spatial_axis=0, prob=prob),
            RandFlipd(keys=keys, spatial_axis=1, prob=prob),
            RandFlipd(keys=keys, spatial_axis=2, prob=prob),
            RandRotate90d(keys=keys, prob=prob, max_k=3),
            RandAffined(
                keys=keys,
                prob=prob,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),  # 5 degrees
                scale_range=(0.05, 0.05, 0.05),
                mode=('bilinear'),
                padding_mode='zeros'
            ),

            # Intensity transformations (only applied to the image)
            RandGaussianNoised(keys=keys, prob=prob * 0.5, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=keys, prob=prob * 0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandAdjustContrastd(keys=keys, prob=prob, gamma=(0.9, 1.1)),
            OneOf([
                RandHistogramShiftd(keys=keys, num_control_points=10, prob=1.0),
                RandBiasFieldd(keys=keys, prob=1.0),
                RandGibbsNoised(keys=keys, prob=1.0, alpha=(0.0, 0.5)),
            ]),

            # Ensure output is tensor
            EnsureTyped(keys=all_keys, data_type="tensor")
        ])

    @staticmethod
    def get_segmentation_transforms(image_keys: KeysCollection, label_keys: KeysCollection, prob: float = 0.5):
        """
        Get transforms for segmentation tasks. Ensures that the same transforms are applied to both
        image and labels where needed.

        Args:
            image_keys (KeysCollection): Keys for image data
            label_keys (KeysCollection): Keys for label/mask data
            prob (float): Probability of applying each transform

        Returns:
            Compose: Composed transforms
        """
        all_keys = []
        if isinstance(image_keys, (str, int)):
            all_keys.append(image_keys)
        else:
            all_keys.extend(list(image_keys))

        if isinstance(label_keys, (str, int)):
            all_keys.append(label_keys)
        else:
            all_keys.extend(list(label_keys))

        return Compose([
            # Spatial transforms applied to both images and labels
            RandFlipd(keys=all_keys, spatial_axis=0, prob=prob),
            RandFlipd(keys=all_keys, spatial_axis=1, prob=prob),
            RandFlipd(keys=all_keys, spatial_axis=2, prob=prob),
            RandRotate90d(keys=all_keys, prob=prob, max_k=3),
            RandAffined(
                keys=all_keys,
                prob=prob,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),  # 5 degrees
                scale_range=(0.05, 0.05, 0.05),
                mode=('bilinear', 'nearest'),  # bilinear for images, nearest for masks
                padding_mode='zeros'
            ),

            # Intensity transforms (only applied to images, not labels)
            RandGaussianNoised(keys=image_keys, prob=prob * 0.5, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=image_keys, prob=prob * 0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5),
                                sigma_z=(0.5, 1.5)),
            RandAdjustContrastd(keys=image_keys, prob=prob, gamma=(0.9, 1.1)),

            # Ensure output is tensor
            EnsureTyped(keys=all_keys, data_type="tensor")
        ])

    @staticmethod
    def get_heavy_augmentation(keys: KeysCollection, prob: float = 0.8):
        """
        Get more aggressive augmentation for scenarios with limited data.

        Args:
            keys (KeysCollection): The keys to apply transforms on
            prob (float): Probability of applying each transform

        Returns:
            Compose: Composed transforms
        """
        return Compose([
            # Spatial transformations
            RandFlipd(keys=keys, spatial_axis=0, prob=prob),
            RandFlipd(keys=keys, spatial_axis=1, prob=prob),
            RandFlipd(keys=keys, spatial_axis=2, prob=prob),
            RandRotate90d(keys=keys, prob=prob, max_k=3),
            RandZoomd(keys=keys, prob=prob, min_zoom=0.9, max_zoom=1.1),
            RandAffined(
                keys=keys,
                prob=prob,
                rotate_range=(np.pi / 20, np.pi / 20, np.pi / 20),  # ~9 degrees
                scale_range=(0.1, 0.1, 0.1),
                mode=('bilinear'),
                padding_mode='zeros',
                translate_range=(10, 10, 10)
            ),

            # Intensity transformations
            RandGaussianNoised(keys=keys, prob=prob, mean=0.0, std=0.2),
            RandGaussianSmoothd(keys=keys, prob=prob, sigma_x=(0.5, 2.0), sigma_y=(0.5, 2.0), sigma_z=(0.5, 2.0)),
            RandAdjustContrastd(keys=keys, prob=prob, gamma=(0.8, 1.2)),
            RandHistogramShiftd(keys=keys, prob=prob, num_control_points=15),
            RandBiasFieldd(keys=keys, prob=prob * 0.8, coeff_range=(0.2, 0.5)),

            # Ensure output is tensor
            ToTensord(keys=keys)
        ])


class AugmentationPipeline:
    """
    Class to create customized augmentation pipelines for neuroimaging data.
    """

    def __init__(self, mode="train", task_type="classification"):
        """
        Initialize the augmentation pipeline.

        Args:
            mode (str): One of 'train', 'val', or 'test'
            task_type (str): One of 'classification', 'segmentation', or 'anomaly'
        """
        self.mode = mode
        self.task_type = task_type

    def get_transforms(self, keys=None, label_keys=None, augment_prob=0.5):
        """
        Get the appropriate transforms based on mode and task type.

        Args:
            keys (KeysCollection): Keys for main data (e.g., 'image')
            label_keys (KeysCollection, optional): Keys for label data
            augment_prob (float): Probability of applying augmentations

        Returns:
            Compose: Composed transforms
        """
        # Default to 'image' key if not provided
        if keys is None:
            keys = ['image']

        # Don't augment validation or test data
        if self.mode in ['val', 'test']:
            return Compose([
                EnsureTyped(keys=keys if label_keys is None else list(keys) + list(label_keys),
                            data_type="tensor")
            ])

        # For training mode, apply appropriate augmentations
        if self.task_type == "classification":
            return AugmentationFactory.get_classification_transforms(
                keys=keys,
                prob=augment_prob,
                addl_keys=label_keys
            )
        elif self.task_type == "segmentation":
            assert label_keys is not None, "Label keys must be provided for segmentation tasks"
            return AugmentationFactory.get_segmentation_transforms(
                image_keys=keys,
                label_keys=label_keys,
                prob=augment_prob
            )
        elif self.task_type == "anomaly":
            # For anomaly detection, use heavier augmentation on the image
            return AugmentationFactory.get_heavy_augmentation(
                keys=keys,
                prob=augment_prob
            )
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


def apply_transforms_to_batch(batch, transforms):
    """
    Apply transforms to a batch of data.

    Args:
        batch (Dict): Batch of data
        transforms (Compose): MONAI transforms to apply

    Returns:
        Dict: Transformed batch
    """
    return transforms(batch)


def get_alzheimers_specific_transforms(keys=None, prob=0.5):
    """
    Get transforms specifically tailored for Alzheimer's disease data.
    These transforms focus on highlighting hippocampal and cortical regions
    that are most affected by AD pathology.

    Args:
        keys (KeysCollection): The keys to apply transforms on
        prob (float): Probability of applying each transform

    Returns:
        Compose: Composed transforms
    """
    if keys is None:
        keys = ['image']

    return Compose([
        # Standard spatial transforms
        RandFlipd(keys=keys, spatial_axis=0, prob=prob),
        RandRotate90d(keys=keys, prob=prob, max_k=3),

        # Moderate affine transforms (avoid extreme distortions for AD analysis)
        RandAffined(
            keys=keys,
            prob=prob,
            rotate_range=(np.pi / 45, np.pi / 45, np.pi / 45),  # ~4 degrees
            scale_range=(0.03, 0.03, 0.03),
            mode=('bilinear'),
            padding_mode='zeros'
        ),

        # Intensity transforms that preserve structural details
        RandGaussianNoised(keys=keys, prob=prob * 0.3, mean=0.0, std=0.05),
        RandAdjustContrastd(keys=keys, prob=prob, gamma=(0.95, 1.05)),

        # Ensure output is tensor
        EnsureTyped(keys=keys, data_type="tensor")
    ])


def get_custom_transforms_for_dataset(dataset_name, keys=None, label_keys=None, prob=0.5):
    """
    Get specialized transforms for specific datasets.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'adni', 'oasis')
        keys (KeysCollection): Keys for main data
        label_keys (KeysCollection, optional): Keys for label data
        prob (float): Probability of applying augmentations

    Returns:
        Compose: Composed transforms
    """
    if keys is None:
        keys = ['image']

    if dataset_name.lower() == 'adni':
        return get_alzheimers_specific_transforms(keys=keys, prob=prob)
    else:
        # Default to standard classification transforms for unknown datasets
        return AugmentationFactory.get_classification_transforms(
            keys=keys,
            prob=prob,
            addl_keys=label_keys
        )