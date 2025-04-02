"""
Brain segmentation model implementation using MONAI components.
This module provides functionality for segmenting brain regions in 3D MRI scans.
"""

import os
import torch
import numpy as np
import nibabel as nib
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('segmentation_model')

# Check which MONAI components are available
try:
    import monai
    from monai.networks.nets import UNet

    # Try to import transforms
    try:
        from monai.transforms import (
            Compose,
            AddChannel,
            ScaleIntensity,
            NormalizeIntensity,
            Orientation,
            Spacing,
            ToTensor
        )
        logger.info("Using MONAI's transforms module")
        MONAI_TRANSFORMS_AVAILABLE = True
    except ImportError:
        logger.warning("MONAI transforms not available, using custom preprocessing")
        MONAI_TRANSFORMS_AVAILABLE = False

    MONAI_AVAILABLE = True
except ImportError:
    logger.error("MONAI not installed, using PyTorch directly")
    MONAI_AVAILABLE = False


class BrainSegmentationModel:
    """
    Class for brain segmentation using MONAI's UNet implementation.
    """

    def __init__(self, model_path=None, device=None, num_classes=4):
        """
        Initialize the brain segmentation model.

        Args:
            model_path (str, optional): Path to a pre-trained model
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
            num_classes (int): Number of segmentation classes (default: 4)
                               Typically: 0=background, 1=grey matter, 2=white matter, 3=csf
        """
        # Metadata
        self.model_creation_date = "2025-04-02 14:07:28"
        self.model_author = "KishoreKumarKalli"

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Define segmentation classes
        self.num_classes = num_classes
        self.class_names = {
            0: "Background",
            1: "Grey Matter",
            2: "White Matter",
            3: "CSF"
        }

        if num_classes > 4:
            self.class_names.update({
                4: "Cerebellum",
                5: "Ventricles",
                6: "Hippocampus",
                7: "Thalamus"
            })

        # Create model architecture
        if MONAI_AVAILABLE:
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(self.device)
        else:
            # Fall back to a basic PyTorch model if MONAI is not available
            from torch import nn
            self.model = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose3d(16, num_classes, kernel_size=2, stride=2),
            ).to(self.device)

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading pre-trained model from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Pre-trained model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading pre-trained model: {str(e)}")
                logger.warning("Continuing with untrained model")
        else:
            logger.warning("No pre-trained model provided. Model will need to be trained or fine-tuned.")

        # Set model to evaluation mode by default
        self.model.eval()

        logger.info(f"Brain segmentation model initialized with {num_classes} classes")

    def preprocess_image(self, image_data):
        """
        Custom preprocessing function that doesn't rely on MONAI transforms.

        Args:
            image_data (numpy.ndarray): Input 3D image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Add channel dimension if needed
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)

        # Normalize intensity
        mean = np.mean(image_data)
        std = np.std(image_data)
        if std > 0:
            image_data = (image_data - mean) / std

        # Convert to torch tensor
        tensor_data = torch.from_numpy(image_data).float().to(self.device)

        return tensor_data

    def train(self, train_files, val_files=None, epochs=100, batch_size=2, learning_rate=1e-4, output_dir="./models"):
        """
        Train the segmentation model.

        Args:
            train_files (list): List of dictionaries with 'image' and 'label' keys for training
            val_files (list, optional): List of dictionaries with 'image' and 'label' keys for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            output_dir (str): Directory to save trained models

        Returns:
            dict: Training history (loss values)
        """
        logger.info(f"Starting model training at: {self.model_creation_date}")
        logger.info(f"Training executed by: {self.model_author}")
        logger.info(f"Training on {len(train_files)} samples for {epochs} epochs")

        # Set model to training mode
        self.model.train()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define loss function and optimizer
        if MONAI_AVAILABLE:
            loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        else:
            # Basic cross entropy loss if MONAI is not available
            loss_function = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [] if val_files else None
        }

        # Custom data loading function
        def load_data_batch(files, indices):
            images = []
            labels = []
            for i in indices:
                if i < len(files):
                    file_info = files[i]
                    # Load image and label
                    image_data = nib.load(file_info['image']).get_fdata()
                    label_data = nib.load(file_info['label']).get_fdata()

                    # Preprocess
                    image_data = self.preprocess_image(image_data).unsqueeze(0)  # Add batch dim
                    label_data = torch.from_numpy(label_data).long().unsqueeze(0).unsqueeze(0).to(self.device)

                    images.append(image_data)
                    labels.append(label_data)

            # Combine into batches
            if images:
                images = torch.cat(images, dim=0)
                labels = torch.cat(labels, dim=0)
                return images, labels
            return None, None

        # Training loop
        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Training
            epoch_loss = 0
            step = 0

            # Process data in batches
            num_samples = len(train_files)
            num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

            for batch_idx in range(num_batches):
                step += 1
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))

                inputs, labels = load_data_batch(train_files, batch_indices)
                if inputs is None:
                    continue

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                logger.debug(f"Epoch {epoch + 1}/{epochs}, Step {step}: Train loss: {loss.item():.4f}")

            # Calculate average loss for the epoch
            if step > 0:
                epoch_loss /= step
                history["train_loss"].append(epoch_loss)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average train loss: {epoch_loss:.4f}")

            # Validation
            if val_files:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_step = 0

                    # Process validation data in batches
                    num_val_samples = len(val_files)
                    num_val_batches = (num_val_samples + batch_size - 1) // batch_size

                    for batch_idx in range(num_val_batches):
                        val_step += 1
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_val_samples)
                        batch_indices = list(range(start_idx, end_idx))

                        val_inputs, val_labels = load_data_batch(val_files, batch_indices)
                        if val_inputs is None:
                            continue

                        val_outputs = self.model(val_inputs)
                        v_loss = loss_function(val_outputs, val_labels)
                        val_loss += v_loss.item()

                    if val_step > 0:
                        val_loss /= val_step
                        history["val_loss"].append(val_loss)
                        logger.info(f"Epoch {epoch + 1}/{epochs}, Average validation loss: {val_loss:.4f}")

                        # Save best model based on validation loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(self.model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                            logger.info(f"Saved new best model at epoch {epoch + 1}")

                self.model.train()

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth"))
                logger.info(f"Saved checkpoint at epoch {epoch + 1}")

        # Save final model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        logger.info("Training completed. Final model saved.")

        # Set model back to evaluation mode
        self.model.eval()

        return history

    def predict(self, image_data, original_img=None):
        """
        Perform segmentation on a single image.

        Args:
            image_data (numpy.ndarray): Input image data (3D)
            original_img (nibabel.nifti1.Nifti1Image, optional): Original NiBabel image object for metadata

        Returns:
            numpy.ndarray: Segmentation mask
        """
        logger.info(f"Performing segmentation at: 2025-04-02 14:07:28")
        logger.info(f"Segmentation executed by: KishoreKumarKalli")

        # Preprocess input data
        input_tensor = self.preprocess_image(image_data).unsqueeze(0)  # Add batch dimension

        # Set model to evaluation mode
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Get the predicted class for each voxel (argmax along the channel dimension)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        logger.info("Segmentation completed successfully")

        return prediction

    def save_segmentation(self, segmentation, original_img, output_path):
        """
        Save segmentation result as a NIfTI file.

        Args:
            segmentation (numpy.ndarray): Segmentation mask
            original_img (nibabel.nifti1.Nifti1Image): Original NiBabel image object
            output_path (str): Path to save the segmentation
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.info(f"Saving segmentation to: {output_path}")

            # Create a new NIfTI image with the segmentation data and original metadata
            seg_img = nib.Nifti1Image(segmentation.astype(np.int16), original_img.affine, original_img.header)
            nib.save(seg_img, output_path)

            logger.info(f"Segmentation saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving segmentation to {output_path}: {str(e)}")
            raise

    def batch_segment(self, input_files, output_dir):
        """
        Perform segmentation on a batch of files.

        Args:
            input_files (list): List of dictionaries with 'image' (path) and 'data' (loaded data) keys
            output_dir (str): Directory to save segmentation results

        Returns:
            list: Paths to segmentation files
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting batch segmentation for {len(input_files)} files")

        output_files = []
        for i, file_info in enumerate(input_files):
            try:
                # Get file information
                file_path = file_info['image']
                image_data = file_info['data'][0]  # Assuming data is a tuple (image_data, image_obj)
                image_obj = file_info['data'][1]

                # Create output file path
                file_name = os.path.basename(file_path)
                output_name = f"seg_{file_name}"
                output_path = os.path.join(output_dir, output_name)

                logger.info(f"Processing file {i + 1}/{len(input_files)}: {file_name}")

                # Perform segmentation
                segmentation = self.predict(image_data, image_obj)

                # Save result
                self.save_segmentation(segmentation, image_obj, output_path)
                output_files.append(output_path)

                logger.info(f"Successfully segmented: {file_name} -> {output_name}")
            except Exception as e:
                logger.error(f"Error segmenting file {file_path}: {str(e)}")
                continue

        logger.info(f"Batch segmentation completed. Processed {len(output_files)} files.")
        return output_files


# Utility functions

def get_segmentation_metrics(prediction, ground_truth, exclude_background=True):
    """
    Calculate segmentation metrics (Dice, IoU) for evaluation.

    Args:
        prediction (numpy.ndarray): Predicted segmentation mask
        ground_truth (numpy.ndarray): Ground truth segmentation mask
        exclude_background (bool): Whether to exclude background (class 0) from metrics

    Returns:
        dict: Dictionary of metrics
    """

    # Convert to one-hot encoding manually if MONAI is not available
    def to_one_hot(arr, num_classes):
        shape = list(arr.shape)
        shape.insert(0, num_classes)
        one_hot = np.zeros(shape, dtype=np.float32)
        for c in range(num_classes):
            one_hot[c][arr == c] = 1
        return one_hot

    # Compute Dice coefficient manually
    def compute_dice(pred, gt, class_idx):
        pred_class = (pred == class_idx).astype(np.float32).flatten()
        gt_class = (gt == class_idx).astype(np.float32).flatten()

        intersection = np.sum(pred_class * gt_class)
        union = np.sum(pred_class) + np.sum(gt_class)

        if union > 0:
            return 2.0 * intersection / union
        return 1.0  # If both prediction and ground truth are empty, dice is 1

    # Get number of classes
    num_classes = max(prediction.max(), ground_truth.max()) + 1

    # Calculate metrics for each class
    metrics = {}
    dice_scores = []

    for i in range(1 if exclude_background else 0, num_classes):
        dice = compute_dice(prediction, ground_truth, i)
        metrics[f"dice_class_{i}"] = dice
        dice_scores.append(dice)

    # Compute mean Dice score
    if dice_scores:
        metrics["mean_dice"] = np.mean(dice_scores)

    return metrics