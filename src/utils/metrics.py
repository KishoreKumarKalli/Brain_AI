import numpy as np
import torch
from scipy import ndimage
import SimpleITK as sitk
from typing import Union, List, Tuple, Dict
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calculate the Dice similarity coefficient between two binary masks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient (float between 0 and 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    return (2. * intersection + smooth) / (union + smooth)


def dice_coefficient_torch(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    PyTorch implementation of Dice coefficient.

    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient as a torch tensor
    """
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)

    return (2. * intersection + smooth) / (union + smooth)


def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray, percentile: int = 95) -> float:
    """
    Calculate the Hausdorff distance between two binary masks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        percentile: Percentile to use (95 for 95% Hausdorff distance)

    Returns:
        Hausdorff distance
    """
    # Convert to binary masks if not already
    y_true_binary = (y_true > 0.5).astype(np.uint8)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)

    # Use SimpleITK for accurate 3D Hausdorff distance
    y_true_sitk = sitk.GetImageFromArray(y_true_binary)
    y_pred_sitk = sitk.GetImageFromArray(y_pred_binary)

    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_filter.Execute(y_true_sitk, y_pred_sitk)

    if percentile == 100:
        return hausdorff_filter.GetHausdorffDistance()
    else:
        # For percentile Hausdorff, we need to calculate distances and take the percentile
        distance_map = sitk.SignedMaurerDistanceMap(y_true_sitk, squaredDistance=False, useImageSpacing=True)
        distance_map_array = sitk.GetArrayFromImage(distance_map)

        # Get distances at the boundaries of the predicted segmentation
        boundary_pts = ndimage.morphology.binary_dilation(y_pred_binary) & (~y_pred_binary)
        if np.sum(boundary_pts) == 0:
            return 0.0

        surface_distances = np.abs(distance_map_array[boundary_pts])
        return np.percentile(surface_distances, percentile)


def volume_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the volumetric similarity between two binary masks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask

    Returns:
        Volumetric similarity (between -1 and 1)
    """
    vol_true = np.sum(y_true)
    vol_pred = np.sum(y_pred)

    if (vol_true + vol_pred) == 0:
        return 1.0  # Both are empty, perfect match

    return 1.0 - abs(vol_true - vol_pred) / (vol_true + vol_pred)


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calculate the sensitivity (recall/true positive rate).

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Sensitivity score (float between 0 and 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_positives = np.sum(y_true_flat * y_pred_flat)
    possible_positives = np.sum(y_true_flat)

    return (true_positives + smooth) / (possible_positives + smooth)


def specificity(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calculate the specificity (true negative rate).

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Specificity score (float between 0 and 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_negatives = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    possible_negatives = np.sum(1 - y_true_flat)

    return (true_negatives + smooth) / (possible_negatives + smooth)


def precision(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calculate the precision (positive predictive value).

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Precision score (float between 0 and 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_positives = np.sum(y_true_flat * y_pred_flat)
    predicted_positives = np.sum(y_pred_flat)

    return (true_positives + smooth) / (predicted_positives + smooth)


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-7) -> float:
    """
    Calculate the Jaccard index (IoU - Intersection over Union).

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score (float between 0 and 1)
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection

    return (intersection + smooth) / (union + smooth)


def calculate_roc_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict:
    """
    Calculate ROC curve and AUC.

    Args:
        y_true: Ground truth binary labels
        y_pred_prob: Predicted probabilities

    Returns:
        Dictionary containing fpr, tpr, and roc_auc
    """
    fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred_prob.flatten())
    roc_auc = auc(fpr, tpr)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'roc_auc': roc_auc
    }


def calculate_precision_recall_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict:
    """
    Calculate precision-recall curve and average precision.

    Args:
        y_true: Ground truth binary labels
        y_pred_prob: Predicted probabilities

    Returns:
        Dictionary containing precision, recall, and average precision
    """
    precision_values, recall_values, thresholds = precision_recall_curve(y_true.flatten(), y_pred_prob.flatten())
    avg_precision = np.mean(precision_values)

    return {
        'precision': precision_values,
        'recall': recall_values,
        'thresholds': thresholds,
        'average_precision': avg_precision
    }


def compute_multi_class_dice(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Compute Dice coefficient for each class in a multi-class segmentation.

    Args:
        y_true: Ground truth segmentation (one-hot or label map)
        y_pred: Predicted segmentation (one-hot or label map)
        num_classes: Number of classes including background

    Returns:
        Dictionary mapping class indices to Dice scores
    """
    dice_scores = {}

    # Convert to one-hot if necessary
    if y_true.shape != y_pred.shape or len(y_true.shape) != len(y_pred.shape) + 1:
        # Assuming y_true and y_pred are label maps
        y_true_one_hot = np.zeros((num_classes,) + y_true.shape, dtype=np.float32)
        y_pred_one_hot = np.zeros((num_classes,) + y_pred.shape, dtype=np.float32)

        for i in range(num_classes):
            y_true_one_hot[i] = (y_true == i).astype(np.float32)
            y_pred_one_hot[i] = (y_pred == i).astype(np.float32)
    else:
        # Already in one-hot format
        y_true_one_hot = y_true
        y_pred_one_hot = y_pred

    # Compute Dice for each class
    for i in range(num_classes):
        dice_scores[i] = dice_coefficient(y_true_one_hot[i], y_pred_one_hot[i])

    return dice_scores


def average_surface_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the average surface distance between two binary masks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask

    Returns:
        Average surface distance
    """
    # Convert to binary masks if not already
    y_true_binary = (y_true > 0.5).astype(np.uint8)
    y_pred_binary = (y_pred > 0.5).astype(np.uint8)

    # Get distance maps
    y_true_distance = ndimage.distance_transform_edt(~y_true_binary)
    y_pred_distance = ndimage.distance_transform_edt(~y_pred_binary)

    # Get the surfaces
    y_true_surface = ndimage.morphology.binary_dilation(y_true_binary) ^ y_true_binary
    y_pred_surface = ndimage.morphology.binary_dilation(y_pred_binary) ^ y_pred_binary

    # Get the surface distances
    y_true_to_pred_distance = y_pred_distance[y_true_surface]
    y_pred_to_true_distance = y_true_distance[y_pred_surface]

    # Combine the distances
    all_distances = np.concatenate([y_true_to_pred_distance, y_pred_to_true_distance])

    # Return the average
    if len(all_distances) > 0:
        return np.mean(all_distances)
    else:
        return 0.0  # No surfaces (both masks are empty or full)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray = None) -> Dict:
    """
    Calculate all relevant segmentation metrics.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        y_pred_prob: Predicted probabilities (optional)

    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {
        'dice': dice_coefficient(y_true, y_pred),
        'jaccard': jaccard_index(y_true, y_pred),
        'sensitivity': sensitivity(y_true, y_pred),
        'specificity': specificity(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'volume_similarity': volume_similarity(y_true, y_pred),
        'hausdorff_distance_95': hausdorff_distance(y_true, y_pred, percentile=95),
        'average_surface_distance': average_surface_distance(y_true, y_pred)
    }

    # Add ROC and PR curves if probabilities are provided
    if y_pred_prob is not None:
        metrics.update({
            'roc_data': calculate_roc_auc(y_true, y_pred_prob),
            'pr_data': calculate_precision_recall_curve(y_true, y_pred_prob)
        })

    return metrics