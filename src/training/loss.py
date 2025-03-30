import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights.

    Args:
        loss_functions (list): List of loss function instances
        weights (list): List of weights for each loss function
    """

    def __init__(self, loss_functions, weights=None):
        super(CombinedLoss, self).__init__()
        self.loss_functions = loss_functions

        if weights is None:
            weights = [1.0] * len(loss_functions)

        assert len(weights) == len(loss_functions), "Number of weights must match number of loss functions"
        self.weights = weights

    def forward(self, pred, target):
        total_loss = 0.0
        for i, loss_fn in enumerate(self.loss_functions):
            loss_value = loss_fn(pred, target)
            total_loss += self.weights[i] * loss_value

        return total_loss


class WeightedDiceLoss(nn.Module):
    """
    Weighted Dice loss for handling class imbalance.

    Args:
        weight (list or torch.Tensor): Weight for each class
        smooth (float): Smoothing factor to prevent division by zero
    """

    def __init__(self, weight=None, smooth=1e-5):
        super(WeightedDiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, pred, target):
        # Get number of classes
        n_classes = pred.size(1)

        # Default weights if not provided
        if self.weight is None:
            weight = torch.ones(n_classes).to(pred.device)
        else:
            weight = self.weight.to(pred.device)

        # Convert to one-hot if target is not one-hot encoded
        if target.dim() == 3 or (target.dim() == 4 and target.size(1) == 1):
            target_one_hot = F.one_hot(target.long().squeeze(1), num_classes=n_classes)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float() if target.dim() == 3 else \
                target_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            target_one_hot = target

        # Compute weighted Dice loss for each class
        dice_loss = 0.0
        for i in range(n_classes):
            dice_loss_i = self._binary_dice_loss(pred[:, i, ...], target_one_hot[:, i, ...])
            dice_loss += weight[i] * dice_loss_i

        return dice_loss / weight.sum()

    def _binary_dice_loss(self, pred, target):
        """Calculate binary Dice loss for a single class."""
        # Flatten predictions and target
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Calculate Dice coefficient and loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that focuses on segmentation borders.

    Args:
        kernel_size (int): Size of the kernel for boundary extraction
        weight (float): Weight for boundary loss
    """

    def __init__(self, kernel_size=3, weight=1.0):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size
        self.weight = weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)

    def forward(self, pred, target):
        # Standard Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Extract boundaries from prediction and target
        pred_boundaries = self._extract_boundaries(pred)
        target_boundaries = self._extract_boundaries(target)

        # Boundary Dice Loss
        boundary_dice_loss = self.dice_loss(pred_boundaries, target_boundaries)

        # Combine losses
        total_loss = dice_loss + self.weight * boundary_dice_loss

        return total_loss

    def _extract_boundaries(self, x):
        """Extract boundaries using gradient operation."""
        # We'll use a simple gradient approach to detect edges
        if x.dim() == 4:  # 2D images batch
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().to(x.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().to(x.device)

            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, x.size(1), 1, 1)
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, x.size(1), 1, 1)

            grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
            grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))

        elif x.dim() == 5:  # 3D volumes batch
            # For 3D, we create simple 3D gradient kernels
            sobel_x = torch.zeros((3, 3, 3)).to(x.device)
            sobel_x[1, 1, 0] = -1
            sobel_x[1, 1, 2] = 1

            sobel_y = torch.zeros((3, 3, 3)).to(x.device)
            sobel_y[1, 0, 1] = -1
            sobel_y[1, 2, 1] = 1

            sobel_z = torch.zeros((3, 3, 3)).to(x.device)
            sobel_z[0, 1, 1] = -1
            sobel_z[2, 1, 1] = 1

            sobel_x = sobel_x.view(1, 1, 3, 3, 3).repeat(1, x.size(1), 1, 1, 1)
            sobel_y = sobel_y.view(1, 1, 3, 3, 3).repeat(1, x.size(1), 1, 1, 1)
            sobel_z = sobel_z.view(1, 1, 3, 3, 3).repeat(1, x.size(1), 1, 1, 1)

            grad_x = F.conv3d(x, sobel_x, padding=1, groups=x.size(1))
            grad_y = F.conv3d(x, sobel_y, padding=1, groups=x.size(1))
            grad_z = F.conv3d(x, sobel_z, padding=1, groups=x.size(1))

            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

        if x.dim() == 4:
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_magnitude


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling small and imbalanced structures in medical images.

    Args:
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        gamma (float): Focal parameter to focus on hard examples
        smooth (float): Smoothing factor
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Get number of classes
        n_classes = pred.size(1)

        # Convert to one-hot if target is not one-hot encoded
        if target.dim() == 3 or (target.dim() == 4 and target.size(1) == 1):
            target_one_hot = F.one_hot(target.long().squeeze(1), num_classes=n_classes)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float() if target.dim() == 3 else \
                target_one_hot.permute(0, 4, 1, 2, 3).float()
        else:
            target_one_hot = target

        # Use sigmoid for binary classification or softmax for multi-class
        if n_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)

        # Compute Focal Tversky loss for each class
        focal_tversky_loss = 0.0

        for i in range(n_classes):
            # Flatten predictions and target for this class
            pred_flat = pred[:, i, ...].view(-1)
            target_flat = target_one_hot[:, i, ...].view(-1)

            # Calculate true positives, false positives, and false negatives
            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()

            # Calculate Tversky index
            tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

            # Apply focal parameter
            focal_tversky = (1 - tversky_index) ** self.gamma

            focal_tversky_loss += focal_tversky

        return focal_tversky_loss / n_classes


class DeepSupervisionLoss(nn.Module):
    """
    Loss function for deep supervision in U-Net like architectures.

    Args:
        main_loss (nn.Module): Main loss function
        aux_weights (list): Weights for auxiliary outputs from different depths
    """

    def __init__(self, main_loss=None, aux_weights=None):
        super(DeepSupervisionLoss, self).__init__()
        self.main_loss = main_loss if main_loss is not None else DiceCELoss(to_onehot_y=True, sigmoid=True)

        # Default weights decrease as we go deeper
        if aux_weights is None:
            self.aux_weights = [0.4, 0.3, 0.2, 0.1]
        else:
            self.aux_weights = aux_weights

    def forward(self, preds, target):
        # If not deep supervision, just return main loss
        if not isinstance(preds, (list, tuple)):
            return self.main_loss(preds, target)

        # Main prediction is the first element
        main_pred = preds[0]
        loss = self.main_loss(main_pred, target)

        # Add losses from auxiliary outputs
        aux_preds = preds[1:]
        num_aux = len(aux_preds)

        # Adjust weights if needed
        weights = self.aux_weights
        if num_aux < len(weights):
            weights = weights[:num_aux]
        elif num_aux > len(weights):
            # Extend with zeros
            weights.extend([0.0] * (num_aux - len(weights)))

        # Apply auxiliary losses with weights
        for i, (aux_pred, weight) in enumerate(zip(aux_preds, weights)):
            # Handle different sizes - resize predictions to match target
            if aux_pred.shape[-2:] != target.shape[-2:]:
                if len(aux_pred.shape) == 4:  # 2D data
                    aux_pred = F.interpolate(aux_pred, target.shape[-2:], mode='bilinear', align_corners=False)
                elif len(aux_pred.shape) == 5:  # 3D data
                    aux_pred = F.interpolate(aux_pred, target.shape[-3:], mode='trilinear', align_corners=False)

            aux_loss = self.main_loss(aux_pred, target)
            loss += weight * aux_loss

        return loss


# Create dictionary of available loss functions for easy selection
LOSS_FUNCTIONS = {
    'dice': DiceLoss(to_onehot_y=True, sigmoid=True),
    'dice_ce': DiceCELoss(to_onehot_y=True, sigmoid=True),
    'focal': FocalLoss(to_onehot_y=True),
    'generalized_dice': GeneralizedDiceLoss(to_onehot_y=True, sigmoid=True),
    'tversky': TverskyLoss(to_onehot_y=True, sigmoid=True),
    'weighted_dice': WeightedDiceLoss(),
    'boundary': BoundaryLoss(),
    'focal_tversky': FocalTverskyLoss(),
}


def get_loss_function(loss_name='dice_ce', **kwargs):
    """
    Get a loss function by name with optional parameters.

    Args:
        loss_name (str): Name of the loss function
        **kwargs: Additional parameters for the loss function

    Returns:
        nn.Module: Loss function
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(f"Loss function '{loss_name}' not found. Available options: {list(LOSS_FUNCTIONS.keys())}")

    if loss_name == 'weighted_dice' and 'weight' in kwargs:
        return WeightedDiceLoss(weight=torch.tensor(kwargs['weight']))

    if loss_name == 'boundary' and ('kernel_size' in kwargs or 'weight' in kwargs):
        kernel_size = kwargs.get('kernel_size', 3)
        weight = kwargs.get('weight', 1.0)
        return BoundaryLoss(kernel_size=kernel_size, weight=weight)

    if loss_name == 'focal_tversky':
        alpha = kwargs.get('alpha', 0.3)
        beta = kwargs.get('beta', 0.7)
        gamma = kwargs.get('gamma', 2.0)
        return FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    # Return the default implementation for other losses
    return LOSS_FUNCTIONS[loss_name]


def create_combined_loss(loss_names=None, weights=None, **kwargs):
    """
    Create a combined loss function from multiple loss functions.

    Args:
        loss_names (list): List of loss function names
        weights (list): List of weights for each loss function
        **kwargs: Additional parameters for loss functions

    Returns:
        CombinedLoss: Combined loss function
    """
    if loss_names is None:
        loss_names = ['dice', 'ce']

    if weights is None:
        weights = [1.0] * len(loss_names)

    loss_functions = [get_loss_function(name, **kwargs) for name in loss_names]

    return CombinedLoss(loss_functions, weights)