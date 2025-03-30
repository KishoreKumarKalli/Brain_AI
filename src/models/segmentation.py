import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet, DynUNet, SegResNet
from monai.networks.blocks import ResidualUnit
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import numpy as np
import os
from pathlib import Path


class BrainSegmentationModel(nn.Module):
    """
    Wrapper class for brain segmentation models that provides
    a consistent interface for different architectures.

    Args:
        model_name (str): Name of the segmentation model architecture
        in_channels (int): Number of input channels
        out_channels (int): Number of output segmentation classes
        dimensions (int): Input data dimensions (2D or 3D)
        features (list): Feature dimensions for each level
        dropout_rate (float): Dropout probability
        use_deep_supervision (bool): Whether to use deep supervision
        pretrained (bool): Whether to load pretrained weights
    """

    def __init__(
            self,
            model_name="unet",
            in_channels=1,
            out_channels=4,  # Background + Gray Matter + White Matter + CSF
            dimensions=3,
            features=None,
            dropout_rate=0.2,
            use_deep_supervision=False,
            pretrained=False,
            checkpoint_path=None
    ):
        super(BrainSegmentationModel, self).__init__()

        self.model_name = model_name.lower()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = dimensions
        self.use_deep_supervision = use_deep_supervision

        # Set default features if not provided
        if features is None:
            self.features = [32, 64, 128, 256, 320]
        else:
            self.features = features

        # Initialize the appropriate model architecture
        if self.model_name == "unet":
            self.model = monai.networks.nets.UNet(
                dimensions=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=self.features,
                strides=[2, 2, 2, 2],
                num_res_units=2,
                dropout=dropout_rate
            )
        elif self.model_name == "dynunet":
            kernel_size = [[3, 3, 3]] * len(self.features)
            strides = [[1, 1, 1]] + [[2, 2, 2]] * (len(self.features) - 1)

            # DynUNet specific parameters
            self.model = monai.networks.nets.DynUNet(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                upsample_kernel_size=strides[1:],
                norm_name=("INSTANCE", {"affine": True}),
                deep_supervision=use_deep_supervision,
                res_block=True
            )
        elif self.model_name == "segresnet":
            self.model = monai.networks.nets.SegResNet(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                init_filters=32,
                dropout_prob=dropout_rate,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
            )
        elif self.model_name == "attentionunet":
            self.model = monai.networks.nets.AttentionUnet(
                spatial_dims=dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=self.features,
                strides=[2, 2, 2, 2],
                dropout=dropout_rate
            )
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W]

        Returns:
            torch.Tensor: Output tensor (and intermediate outputs if deep_supervision=True)
        """
        return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            # Handle case where the model was saved with nn.DataParallel
            state_dict = checkpoint["state_dict"]
            if list(state_dict.keys())[0].startswith("module."):
                # Remove the 'module.' prefix
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            # Assume the checkpoint is just the state dict
            self.load_state_dict(checkpoint)

    def save_checkpoint(self, checkpoint_path, optimizer=None, epoch=None, best_metric=None):
        """
        Save model checkpoint.

        Args:
            checkpoint_path (str): Path to save the checkpoint
            optimizer (torch.optim.Optimizer, optional): Optimizer state
            epoch (int, optional): Current epoch
            best_metric (float, optional): Best metric value
        """
        Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "dimensions": self.dimensions,
            "features": self.features,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if best_metric is not None:
            checkpoint["best_metric"] = best_metric

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def get_parameters(self):
        """Get model parameters for optimization."""
        return self.parameters()

    def get_backbone_parameters(self):
        """Get backbone parameters (useful for fine-tuning)."""
        return self.model.parameters()


class BrainSegmentationEnsemble(nn.Module):
    """
    Ensemble of multiple segmentation models for more robust predictions.

    Args:
        models (list): List of BrainSegmentationModel instances
        method (str): Ensemble method ('average', 'majority', or 'weighted')
        weights (list, optional): Weights for each model if using 'weighted' method
    """

    def __init__(self, models, method="average", weights=None):
        super(BrainSegmentationEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.method = method

        # Set equal weights if not provided
        if weights is None:
            self.weights = torch.ones(self.num_models) / self.num_models
        else:
            assert len(weights) == self.num_models, "Number of weights must match number of models"
            self.weights = torch.tensor(weights) / sum(weights)

    def forward(self, x):
        """
        Forward pass through all models and combine their predictions.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Ensemble prediction
        """
        # Get predictions from all models
        predictions = [model(x) for model in self.models]

        # Ensure all predictions are processed for proper ensembling
        processed_preds = []
        for pred in predictions:
            # Handle deep supervision outputs
            if isinstance(pred, tuple):
                # Only use final output for ensembling
                pred = pred[0]
            processed_preds.append(pred)

        # Combine predictions based on the chosen method
        if self.method == "average":
            # Simple average of softmax probabilities
            softmax_preds = [F.softmax(p, dim=1) for p in processed_preds]
            ensemble_pred = torch.zeros_like(softmax_preds[0])
            for pred in softmax_preds:
                ensemble_pred += pred / self.num_models

        elif self.method == "majority":
            # Hard voting (majority)
            preds = [torch.argmax(p, dim=1, keepdim=True) for p in processed_preds]
            stacked = torch.cat(preds, dim=1)

            # Count occurrences of each class for each voxel
            ensemble_pred = torch.zeros_like(processed_preds[0])
            for c in range(ensemble_pred.size(1)):
                votes = (stacked == c).sum(dim=1, keepdim=True).float()
                ensemble_pred[:, c:c + 1] = votes

            # Normalize to get probabilities
            ensemble_pred = ensemble_pred / self.num_models

        elif self.method == "weighted":
            # Weighted average of softmax probabilities
            softmax_preds = [F.softmax(p, dim=1) for p in processed_preds]
            ensemble_pred = torch.zeros_like(softmax_preds[0])
            for i, pred in enumerate(softmax_preds):
                ensemble_pred += pred * self.weights[i].to(x.device)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")

        # Convert probabilities back to logits for loss functions
        ensemble_pred = torch.log(ensemble_pred + 1e-8)

        return ensemble_pred

    def set_weights(self, new_weights):
        """Update ensemble weights."""
        assert len(new_weights) == self.num_models, "Number of weights must match number of models"
        self.weights = torch.tensor(new_weights) / sum(new_weights)

    def get_parameters(self):
        """Get parameters of all models in the ensemble."""
        return self.parameters()


class BrainAnomalyDetector(nn.Module):
    """
    Model for detecting anomalies in brain MRI scans.
    Uses a U-Net-based architecture with a decoder branch for reconstruction
    and compares input with reconstruction to detect anomalies.

    Args:
        in_channels (int): Number of input channels
        features (list): Feature dimensions for each level
        latent_dim (int): Size of the latent dimension for bottleneck
        dropout_rate (float): Dropout probability
    """

    def __init__(
            self,
            in_channels=1,
            features=None,
            latent_dim=128,
            dropout_rate=0.2
    ):
        super(BrainAnomalyDetector, self).__init__()

        if features is None:
            self.features = [32, 64, 128, 256, 512]
        else:
            self.features = features

        # Encoder
        self.encoder = nn.ModuleList()
        in_features = in_channels
        for feature in self.features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_features, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(feature, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
            )
            in_features = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.features[-1], latent_dim, kernel_size=1),
            nn.InstanceNorm3d(latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(latent_dim, self.features[-1], kernel_size=1),
            nn.InstanceNorm3d(self.features[-1]),
            nn.LeakyReLU(inplace=True)
        )

        # Decoder for reconstruction
        self.decoder = nn.ModuleList()
        for feature in reversed(self.features):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2),
                    nn.Conv3d(feature, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(feature, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True)
                )
            )

        # Final layer to reconstruct the input
        self.final = nn.Conv3d(self.features[0], in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            features.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            skip_features = features[-(i + 1)]

            # Ensure dimensions match
            if x.shape[2:] != skip_features.shape[2:]:
                x = F.interpolate(x, size=skip_features.shape[2:], mode='trilinear', align_corners=False)

            # Concatenate skip connection
            x = torch.cat((skip_features, x), dim=1)
            x = decoder_block(x)

        # Final convolution
        reconstruction = self.final(x)

        # Calculate residual (anomaly map)
        anomaly_map = torch.abs(x - reconstruction)

        return reconstruction, anomaly_map

    def get_parameters(self):
        """Get model parameters for optimization."""
        return self.parameters()


def get_loss_function(loss_name="dice", lambda_ce=0.5, lambda_dice=0.5, include_background=False):
    """
    Get the specified loss function.

    Args:
        loss_name (str): Name of the loss function
        lambda_ce (float): Weight of cross-entropy loss in combined losses
        lambda_dice (float): Weight of Dice loss in combined losses
        include_background (bool): Whether to include background class in loss calculation

    Returns:
        nn.Module: Loss function
    """
    if loss_name.lower() == "dice":
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            reduction="mean"
        )
    elif loss_name.lower() == "dicece":
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=True,
            softmax=True,
            lambda_ce=lambda_ce,
            lambda_dice=lambda_dice
        )
    elif loss_name.lower() == "focal":
        return FocalLoss(
            include_background=include_background,
            to_onehot_y=True,
            gamma=2.0
        )
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def get_segmentation_model(model_name="unet", in_channels=1, out_channels=4, dimensions=3,
                           pretrained=False, checkpoint_path=None):
    """
    Factory function to create a segmentation model.

    Args:
        model_name (str): Name of the model architecture
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dimensions (int): Input dimensions (2D or 3D)
        pretrained (bool): Whether to use pretrained weights
        checkpoint_path (str, optional): Path to checkpoint file

    Returns:
        nn.Module: Instantiated model
    """
    if model_name.lower() == "ensemble":
        # Create an ensemble of models
        models = [
            BrainSegmentationModel("unet", in_channels, out_channels, dimensions),
            BrainSegmentationModel("segresnet", in_channels, out_channels, dimensions),
            BrainSegmentationModel("attentionunet", in_channels, out_channels, dimensions)
        ]

        # Load individual checkpoints if available
        if checkpoint_path and isinstance(checkpoint_path, list):
            for i, cp in enumerate(checkpoint_path):
                if cp and os.path.exists(cp):
                    models[i].load_checkpoint(cp)

        return BrainSegmentationEnsemble(models, method="weighted")

    elif model_name.lower() == "anomaly":
        return BrainAnomalyDetector(in_channels=in_channels)

    else:
        return BrainSegmentationModel(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=out_channels,
            dimensions=dimensions,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path
        )