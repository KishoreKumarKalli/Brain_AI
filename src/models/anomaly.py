import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution
import os
from pathlib import Path


class EncoderBlock(nn.Module):
    """
    Encoder block for the brain anomaly detection model.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        padding (int): Padding for convolutions
        dropout_p (float): Dropout probability
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.2):
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # First convolution block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Save features before pooling for skip connections
        features = x

        # Apply pooling
        x = self.pool(x)

        return x, features


class DecoderBlock(nn.Module):
    """
    Decoder block for the brain anomaly detection model.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for convolutions
        stride (int): Stride for convolutions
        padding (int): Padding for convolutions
        dropout_p (float): Dropout probability
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_p=0.2):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_p)

    def forward(self, x, skip_features):
        # Upsample
        x = self.upconv(x)

        # Handle the case where dimensions don't match exactly
        if x.shape[2:] != skip_features.shape[2:]:
            x = F.interpolate(x, size=skip_features.shape[2:], mode='trilinear', align_corners=False)

        # Concatenate with skip connection
        x = torch.cat([x, skip_features], dim=1)

        # First convolution block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x


class BrainAnomalyDetectionModel(nn.Module):
    """
    A model for detecting anomalies in brain MRI images using
    a reconstruction-based approach with U-Net architecture.

    Args:
        in_channels (int): Number of input channels
        features (list): List of feature dimensions for each level
        dropout_p (float): Dropout probability
        latent_dim (int): Dimension of the latent space
    """

    def __init__(self, in_channels=1, features=None, dropout_p=0.2, latent_dim=256):
        super(BrainAnomalyDetectionModel, self).__init__()

        if features is None:
            self.features = [32, 64, 128, 256, 512]
        else:
            self.features = features

        # Encoder pathway
        self.encoder_blocks = nn.ModuleList()
        in_channels_encoder = in_channels
        for feature in self.features:
            self.encoder_blocks.append(EncoderBlock(in_channels_encoder, feature, dropout_p=dropout_p))
            in_channels_encoder = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(self.features[-1], latent_dim, kernel_size=1),
            nn.InstanceNorm3d(latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(latent_dim, self.features[-1], kernel_size=1),
            nn.InstanceNorm3d(self.features[-1]),
            nn.LeakyReLU(inplace=True),
        )

        # Decoder pathway
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(self.features) - 1, 0, -1):
            # Input channels is current feature + previous feature (skip connection)
            in_channels_decoder = self.features[i] + self.features[i - 1]
            out_channels_decoder = self.features[i - 1]
            self.decoder_blocks.append(
                DecoderBlock(in_channels_decoder, out_channels_decoder, dropout_p=dropout_p)
            )

        # Final reconstruction layer
        self.final_conv = nn.Conv3d(self.features[0], in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pathway and store skip connections
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x, features = encoder_block(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pathway with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = len(skip_connections) - 1 - i
            x = decoder_block(x, skip_connections[skip_idx])

        # Final convolution
        reconstruction = self.final_conv(x)

        # Calculate residual (anomaly map)
        input_tensor = x
        anomaly_map = torch.abs(input_tensor - reconstruction)

        return reconstruction, anomaly_map


class VAEAnomalyDetector(nn.Module):
    """
    Variational Autoencoder for anomaly detection in brain MRI.
    This model learns to encode brain MRIs into a latent space and decode them,
    detecting anomalies as regions with high reconstruction error.

    Args:
        in_channels (int): Number of input channels
        features (list): List of feature dimensions for each level
        latent_dim (int): Dimension of the latent space
        dropout_p (float): Dropout probability
    """

    def __init__(self, in_channels=1, features=None, latent_dim=256, dropout_p=0.2):
        super(VAEAnomalyDetector, self).__init__()

        if features is None:
            self.features = [32, 64, 128, 256, 512]
        else:
            self.features = features

        self.latent_dim = latent_dim

        # Encoder pathway
        self.encoder = nn.ModuleList()
        in_channels_encoder = in_channels
        for feature in self.features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(in_channels_encoder, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(feature, feature, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(feature),
                    nn.LeakyReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
            )
            in_channels_encoder = feature

        # Compute the final encoded feature map dimensions
        # For a 128^3 input and 5 pooling layers, this will be 4^3
        self.encoded_shape = (self.features[-1], 4, 4, 4)
        self.encoded_size = np.prod(self.encoded_shape)

        # Latent space projections
        self.fc_mu = nn.Linear(self.encoded_size, self.latent_dim)
        self.fc_var = nn.Linear(self.encoded_size, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.encoded_size)

        # Decoder pathway
        self.decoder = nn.ModuleList()
        for i in range(len(self.features) - 1, -1, -1):
            if i == len(self.features) - 1:
                # First decoder block takes bottleneck features
                in_channels_decoder = self.features[i]
            else:
                # Other decoder blocks take upsampled features
                in_channels_decoder = self.features[i + 1]

            if i == 0:
                # Final decoder block outputs reconstruction
                out_channels_decoder = in_channels
            else:
                out_channels_decoder = self.features[i]

            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels_decoder, out_channels_decoder, kernel_size=2, stride=2),
                    nn.Conv3d(out_channels_decoder, out_channels_decoder, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_channels_decoder),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(out_channels_decoder, out_channels_decoder, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_channels_decoder),
                    nn.LeakyReLU(inplace=True)
                )
            )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            logvar (torch.Tensor): Log variance of the latent Gaussian

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Get batch size for reshaping later
        batch_size = x.shape[0]

        # Encoder pathway
        for encoder_block in self.encoder:
            x = encoder_block(x)

        # Flatten for latent space projection
        x_flattened = x.view(batch_size, -1)

        # Get latent parameters and sample latent vector
        mu = self.fc_mu(x_flattened)
        logvar = self.fc_var(x_flattened)
        z = self.reparameterize(mu, logvar)

        # Decode latent vector
        x_decoded = self.fc_decode(z)
        x_decoded = x_decoded.view(batch_size, *self.encoded_shape)

        # Decoder pathway
        for decoder_block in self.decoder:
            x_decoded = decoder_block(x_decoded)

        # Calculate residual (anomaly map)
        anomaly_map = torch.abs(x - x_decoded)

        return x_decoded, anomaly_map, mu, logvar


def get_anomaly_model(model_type="autoencoder", in_channels=1, features=None, latent_dim=256):
    """
    Factory function to get the specified anomaly detection model.

    Args:
        model_type (str): Type of anomaly model ('autoencoder' or 'vae')
        in_channels (int): Number of input channels
        features (list): Feature dimensions for each level
        latent_dim (int): Dimension of the latent space

    Returns:
        nn.Module: The anomaly detection model
    """
    if features is None:
        features = [32, 64, 128, 256, 512]

    if model_type.lower() == "autoencoder":
        return BrainAnomalyDetectionModel(
            in_channels=in_channels,
            features=features,
            latent_dim=latent_dim
        )
    elif model_type.lower() == "vae":
        return VAEAnomalyDetector(
            in_channels=in_channels,
            features=features,
            latent_dim=latent_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

