import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Act, Norm
import numpy as np


class ConvBlock(nn.Module):
    """
    A convolutional block with two convolutional layers, normalization, and activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        padding (int): Padding size
        dropout_p (float): Dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dropout_p=0.0
    ):
        super(ConvBlock, self).__init__()

        self.conv1 = Convolution(
            dimensions=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=dropout_p,
        )

        self.conv2 = Convolution(
            dimensions=3,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=dropout_p,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net architecture.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        padding (int): Padding size
        dropout_p (float): Dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dropout_p=0.0
    ):
        super(DownBlock, self).__init__()

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net architecture.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        padding (int): Padding size
        bilinear (bool): Whether to use bilinear interpolation for upsampling
        dropout_p (float): Dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bilinear=True,
            dropout_p=0.0
    ):
        super(UpBlock, self).__init__()

        # If bilinear, use bilinear interpolation and reduce number of channels by half
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv_block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dropout_p=dropout_p,
            )
        else:
            # Otherwise use transposed convolution
            self.up = UpSample(
                dimensions=3,
                in_channels=in_channels,
                out_channels=in_channels // 2,
                scale_factor=2,
            )
            self.conv_block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dropout_p=dropout_p,
            )

    def forward(self, x, skip_connection):
        x = self.up(x)

        # Handle the case where dimensions don't match
        diff_z = skip_connection.size()[2] - x.size()[2]
        diff_y = skip_connection.size()[3] - x.size()[3]
        diff_x = skip_connection.size()[4] - x.size()[4]

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2,
                      diff_z // 2, diff_z - diff_z // 2])

        # Concatenate along the channel dimension
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv_block(x)

        return x


class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant regions in feature maps.

    Args:
        F_g (int): Number of channels in gating signal
        F_l (int): Number of channels in input feature map
        F_int (int): Number of channels for attention computation
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet3D(nn.Module):
    """
    3D U-Net architecture with optional attention gates.

    Args:
        in_channels (int): Number of input channels (default: 1 for single MRI volume)
        out_channels (int): Number of output channels/classes
        features (list): List of feature dimensions for each level
        dropout_p (float): Dropout probability
        use_attention (bool): Whether to use attention gates
        bilinear (bool): Whether to use bilinear upsampling
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256, 512],
            dropout_p=0.0,
            use_attention=False,
            bilinear=True
    ):
        super(UNet3D, self).__init__()

        self.use_attention = use_attention
        self.bilinear = bilinear

        # Initial convolution block
        self.initial = ConvBlock(
            in_channels=in_channels,
            out_channels=features[0],
            dropout_p=dropout_p
        )

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(
                DownBlock(
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    dropout_p=dropout_p
                )
            )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(
                    in_channels=features[i] + features[i - 1] if i == len(features) - 1 or not bilinear else features[
                        i],
                    out_channels=features[i - 1],
                    bilinear=bilinear,
                    dropout_p=dropout_p
                )
            )

        # Attention gates
        if self.use_attention:
            self.attention_gates = nn.ModuleList()
            for i in range(len(features) - 1, 0, -1):
                self.attention_gates.append(
                    AttentionGate(
                        F_g=features[i],
                        F_l=features[i - 1],
                        F_int=features[i - 1] // 2
                    )
                )

        # Final layer to produce output
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Initial feature extraction
        x0 = self.initial(x)

        # Downsampling and storing skip connections
        skip_connections = [x0]
        xi = x0

        for i, down in enumerate(self.down_blocks):
            xi = down(xi)
            if i < len(self.down_blocks) - 1:  # Don't add bottleneck to skip connections
                skip_connections.append(xi)

        # Upsampling and concatenating with skip connections
        for i, up in enumerate(self.up_blocks):
            skip_idx = len(skip_connections) - i - 1
            skip = skip_connections[skip_idx]

            # Apply attention if enabled
            if self.use_attention:
                skip = self.attention_gates[i](xi, skip)

            xi = up(xi, skip)

        # Final 1x1 convolution to produce output channels
        output = self.final(xi)

        return output


class UNet3DWithClassifier(nn.Module):
    """
    3D U-Net with an additional classifier head for diagnosis prediction.

    Args:
        in_channels (int): Number of input channels
        seg_out_channels (int): Number of segmentation output channels/classes
        cls_out_channels (int): Number of classification output channels/classes
        features (list): List of feature dimensions for each level
        dropout_p (float): Dropout probability
        use_attention (bool): Whether to use attention gates
        bilinear (bool): Whether to use bilinear upsampling
    """

    def __init__(
            self,
            in_channels=1,
            seg_out_channels=1,
            cls_out_channels=3,  # For CN, MCI, AD
            features=[32, 64, 128, 256, 512],
            dropout_p=0.0,
            use_attention=True,
            bilinear=True
    ):
        super(UNet3DWithClassifier, self).__init__()

        # Create the U-Net backbone
        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=seg_out_channels,
            features=features,
            dropout_p=dropout_p,
            use_attention=use_attention,
            bilinear=bilinear
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, cls_out_channels)
        )

        # We'll get the bottleneck features for classification
        self.bottleneck_idx = len(features) - 1

    def forward(self, x):
        # Initial feature extraction
        x0 = self.unet.initial(x)

        # Downsampling and storing skip connections
        skip_connections = [x0]
        xi = x0

        for i, down in enumerate(self.unet.down_blocks):
            xi = down(xi)
            if i < len(self.unet.down_blocks) - 1:
                skip_connections.append(xi)

        # Store bottleneck features for classification
        bottleneck = xi

        # Continue with upsampling for segmentation
        for i, up in enumerate(self.unet.up_blocks):
            skip_idx = len(skip_connections) - i - 1
            skip = skip_connections[skip_idx]

            if self.unet.use_attention:
                skip = self.unet.attention_gates[i](xi, skip)

            xi = up(xi, skip)

        # Final 1x1 convolution for segmentation
        seg_output = self.unet.final(xi)

        # Classification from bottleneck features
        cls_output = self.classifier(bottleneck)

        return seg_output, cls_output


def get_model(model_name, in_channels=1, out_channels=1, num_classes=3, features=None):
    """
    Factory function to get the specified model.

    Args:
        model_name (str): Name of the model ('unet3d' or 'unet3d_classifier')
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels for segmentation
        num_classes (int): Number of classes for classification
        features (list, optional): Feature dimensions for each level

    Returns:
        nn.Module: The specified model
    """
    if features is None:
        features = [32, 64, 128, 256, 512]

    if model_name.lower() == 'unet3d':
        return UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            use_attention=True
        )
    elif model_name.lower() == 'unet3d_classifier':
        return UNet3DWithClassifier(
            in_channels=in_channels,
            seg_out_channels=out_channels,
            cls_out_channels=num_classes,
            features=features,
            use_attention=True
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")