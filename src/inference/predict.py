import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import monai
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    ToTensord,
    AsDiscreted
)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.segmentation import BrainSegmentationModel
from utils.metrics import compute_metrics


class BrainSegmentationPredictor:
    """
    Class for inference with brain segmentation models.

    Args:
        model_path (str): Path to the trained model checkpoint
        config (dict): Configuration dictionary
        device (torch.device): Device to use for inference
    """

    def __init__(self, model_path, config=None, device=None):
        self.model_path = model_path
        self.config = config if config is not None else self._load_config_from_checkpoint()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Set up the inference transforms
        self.transforms = self._get_transforms()

        # Set up post-processing
        self.post_process = AsDiscreted(keys="pred", argmax=True)

        # Configure sliding window inference
        self.roi_size = self.config.get('roi_size', [96, 96, 96])
        self.sw_batch_size = self.config.get('sw_batch_size', 4)
        self.overlap = self.config.get('overlap', 0.5)

    def _load_config_from_checkpoint(self):
        """Load configuration from the checkpoint file."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            print("Warning: No configuration found in checkpoint. Using default values.")
            return {}

    def _load_model(self):
        """Load the model from checkpoint."""
        # Create model based on configuration
        model = BrainSegmentationModel(
            model_type=self.config.get('model_type', 'unet'),
            in_channels=self.config.get('in_channels', 1),
            out_channels=self.config.get('num_classes', 2),
            dimensions=self.config.get('dimensions', 3),
            features=self.config.get('features', [32, 64, 128, 256, 320]),
            dropout_prob=self.config.get('dropout_prob', 0.1)
        )

        # Load weights from checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Model loaded from {self.model_path}")
        return model

    def _get_transforms(self):
        """Get transforms for inference."""
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=self.config.get('spacing', [1.0, 1.0, 1.0]),
                mode=["bilinear"]
            ),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image"])
        ])

    def predict_single_image(self, image_path, output_path=None, return_probabilities=False):
        """
        Predict segmentation for a single image.

        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the segmentation result
            return_probabilities (bool): Whether to return probability maps

        Returns:
            tuple: (segmentation, probabilities, original_image)
        """
        # Prepare input
        data = {"image": image_path}
        data = self.transforms(data)
        input_tensor = data["image"].unsqueeze(0).to(self.device)

        # Get original image metadata for saving
        original_img = nib.load(image_path)
        original_affine = original_img.affine
        original_header = original_img.header

        with torch.no_grad():
            # Use sliding window inference for 3D volumes
            prob_map = sliding_window_inference(
                inputs=input_tensor,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.overlap
            )

            # Post-processing
            if return_probabilities:
                # Apply softmax to get probabilities
                prob_map = torch.softmax(prob_map, dim=1)

            # Get segmentation mask
            segmentation = self.post_process({"pred": prob_map})["pred"]

        # Convert to numpy arrays
        segmentation_np = segmentation.squeeze(0).cpu().numpy()
        prob_map_np = prob_map.squeeze(0).cpu().numpy() if return_probabilities else None
        original_img_np = data["image"].squeeze(0).cpu().numpy()

        # Save segmentation if output path is provided
        if output_path is not None:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as NIfTI with original metadata
            seg_nifti = nib.Nifti1Image(segmentation_np.astype(np.uint8), original_affine, header=original_header)
            nib.save(seg_nifti, output_path)

            print(f"Segmentation saved to {output_path}")

        return segmentation_np, prob_map_np, original_img_np

        def predict_batch(self, image_paths, output_dir=None, return_probabilities=False):
            """
            Predict segmentation for a batch of images.

            Args:
                image_paths (list): List of paths to input images
                output_dir (str, optional): Directory to save segmentation results
                return_probabilities (bool): Whether to return probability maps

            Returns:
                dict: Dictionary of prediction results
            """
            results = {}

            for image_path in tqdm(image_paths, desc="Processing images"):
                # Get subject ID from filename
                subject_id = Path(image_path).stem

                # Define output path if needed
                output_path = None
                if output_dir is not None:
                    output_path = os.path.join(output_dir, f"{subject_id}_seg.nii.gz")

                # Predict segmentation
                segmentation, prob_map, original_img = self.predict_single_image(
                    image_path,
                    output_path,
                    return_probabilities
                )

                # Store results
                results[subject_id] = {
                    "segmentation": segmentation,
                    "original_image": original_img
                }

                if return_probabilities:
                    results[subject_id]["probabilities"] = prob_map

            return results

        def compute_volumetric_statistics(self, segmentations_dir, clinical_data_path=None):
            """
            Compute volumetric statistics for segmented regions.

            Args:
                segmentations_dir (str): Directory containing segmentation files
                clinical_data_path (str, optional): Path to clinical data CSV

            Returns:
                pd.DataFrame: DataFrame with volumetric statistics
            """
            # Get list of segmentation files
            seg_files = [f for f in os.listdir(segmentations_dir) if f.endswith('.nii.gz')]

            # Load clinical data if provided
            clinical_data = None
            if clinical_data_path is not None:
                clinical_data = pd.read_csv(clinical_data_path)

            # Prepare data structure for results
            results = []

            # Iterate through segmentation files
            for seg_file in tqdm(seg_files, desc="Computing statistics"):
                # Load segmentation
                seg_path = os.path.join(segmentations_dir, seg_file)
                seg_nifti = nib.load(seg_path)
                seg_data = seg_nifti.get_fdata()

                # Get subject ID from filename
                subject_id = Path(seg_file).stem
                if subject_id.endswith("_seg"):
                    subject_id = subject_id[:-4]  # Remove '_seg' suffix

                # Get voxel volume in mm³
                voxel_size = np.prod(seg_nifti.header.get_zooms())

                # Calculate volume for each class
                volumes = {}
                for class_idx in range(1, self.config.get('num_classes', 2)):
                    # Count voxels for this class and convert to volume
                    class_volume_mm3 = np.sum(seg_data == class_idx) * voxel_size
                    volumes[f"class_{class_idx}_volume_mm3"] = class_volume_mm3

                # Create result entry
                result = {
                    "subject_id": subject_id,
                    **volumes
                }

                # Add clinical data if available
                if clinical_data is not None:
                    subject_clinical = clinical_data[clinical_data['PTID'] == subject_id]
                    if not subject_clinical.empty:
                        for col in clinical_data.columns:
                            if col != 'PTID':  # Skip ID column
                                result[col] = subject_clinical.iloc[0][col]

                results.append(result)

            # Convert to DataFrame
            return pd.DataFrame(results)

        def detect_anomalies(self, input_dir, output_dir=None, threshold=0.8):
            """
            Detect anomalies in brain MRI images.

            Args:
                input_dir (str): Directory containing input images
                output_dir (str, optional): Directory to save anomaly detection results
                threshold (float): Threshold for anomaly detection

            Returns:
                dict: Dictionary of anomaly detection results
            """
            # Get list of input files
            input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                           if f.endswith('.nii') or f.endswith('.nii.gz')]

            results = {}

            for input_file in tqdm(input_files, desc="Detecting anomalies"):
                # Get subject ID from filename
                subject_id = Path(input_file).stem

                # Predict with probability maps
                _, prob_map, original_img = self.predict_single_image(
                    input_file,
                    return_probabilities=True
                )

                # Identify low confidence regions as potential anomalies
                # Use maximum class probability as confidence measure
                if prob_map is not None:
                    confidence_map = np.max(prob_map, axis=0)
                    anomaly_map = 1.0 - confidence_map  # Invert confidence to get anomaly score
                    anomaly_binary = anomaly_map > (1.0 - threshold)  # Threshold to binary anomaly mask

                    # Store results
                    results[subject_id] = {
                        "anomaly_score": anomaly_map,
                        "anomaly_binary": anomaly_binary,
                        "original_image": original_img
                    }

                    # Save anomaly map if output directory is provided
                    if output_dir is not None:
                        # Create output directory if it doesn't exist
                        os.makedirs(output_dir, exist_ok=True)

                        # Load original image to get affine and header
                        original_nifti = nib.load(input_file)

                        # Save anomaly score map
                        anomaly_score_path = os.path.join(output_dir, f"{subject_id}_anomaly_score.nii.gz")
                        anomaly_score_nifti = nib.Nifti1Image(
                            anomaly_map.astype(np.float32),
                            original_nifti.affine,
                            header=original_nifti.header
                        )
                        nib.save(anomaly_score_nifti, anomaly_score_path)

                        # Save binary anomaly map
                        anomaly_binary_path = os.path.join(output_dir, f"{subject_id}_anomaly_binary.nii.gz")
                        anomaly_binary_nifti = nib.Nifti1Image(
                            anomaly_binary.astype(np.uint8),
                            original_nifti.affine,
                            header=original_nifti.header
                        )
                        nib.save(anomaly_binary_nifti, anomaly_binary_path)

            return results

        def evaluate_predictions(self, prediction_dir, ground_truth_dir):
            """
            Evaluate predictions against ground truth.

            Args:
                prediction_dir (str): Directory containing prediction files
                ground_truth_dir (str): Directory containing ground truth files

            Returns:
                pd.DataFrame: DataFrame with evaluation metrics
            """
            # Get list of prediction files
            pred_files = [f for f in os.listdir(prediction_dir) if f.endswith('.nii.gz')]

            # Prepare data structure for results
            results = []

            # Iterate through prediction files
            for pred_file in tqdm(pred_files, desc="Evaluating predictions"):
                # Get subject ID from filename
                subject_id = Path(pred_file).stem
                if subject_id.endswith("_seg"):
                    subject_id = subject_id[:-4]  # Remove '_seg' suffix

                # Find corresponding ground truth file
                gt_file = f"{subject_id}.nii.gz"  # Adjust naming convention as needed
                gt_path = os.path.join(ground_truth_dir, gt_file)

                if not os.path.exists(gt_path):
                    print(f"Ground truth not found for {subject_id}, skipping...")
                    continue

                # Load prediction and ground truth
                pred_nifti = nib.load(os.path.join(prediction_dir, pred_file))
                gt_nifti = nib.load(gt_path)

                pred_data = pred_nifti.get_fdata().astype(np.int64)
                gt_data = gt_nifti.get_fdata().astype(np.int64)

                # Ensure same shape
                if pred_data.shape != gt_data.shape:
                    print(f"Shape mismatch for {subject_id}, skipping...")
                    continue

                # Compute metrics
                metrics = compute_metrics(pred_data, gt_data, n_classes=self.config.get('num_classes', 2))

                # Add subject ID to metrics
                metrics['subject_id'] = subject_id
                results.append(metrics)

            # Convert to DataFrame
            return pd.DataFrame(results)

        def predict_and_visualize(self, image_path, output_path=None, show=True):
            """
            Predict segmentation and visualize results.

            Args:
                image_path (str): Path to input image
                output_path (str, optional): Path to save visualization
                show (bool): Whether to display visualization

            Returns:
                tuple: (segmentation, original_image)
            """
            # Predict segmentation
            segmentation, _, original_img = self.predict_single_image(image_path)

            # Create visualization
            fig = plt.figure(figsize=(15, 5))

            # Get middle slices
            z_mid = segmentation.shape[2] // 2

            # Plot original image
            plt.subplot(1, 3, 1)
            plt.title("Original")
            plt.imshow(original_img[:, :, z_mid], cmap='gray')
            plt.axis('off')

            # Plot segmentation
            plt.subplot(1, 3, 2)
            plt.title("Segmentation")
            plt.imshow(segmentation[:, :, z_mid])
            plt.axis('off')

            # Plot overlay
            plt.subplot(1, 3, 3)
            plt.title("Overlay")
            plt.imshow(original_img[:, :, z_mid], cmap='gray')
            plt.imshow(segmentation[:, :, z_mid], alpha=0.5)
            plt.axis('off')

            plt.tight_layout()

            # Save visualization if output path is provided
            if output_path is not None:
                plt.savefig(output_path)
                print(f"Visualization saved to {output_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()

            return segmentation, original_img

    def batch_inference(config, output_dir):
        """
        Run batch inference on a set of images.

        Args:
            config (dict): Configuration dictionary
            output_dir (str): Directory to save results

        Returns:
            dict: Dictionary of prediction results
        """
        # Create predictor
        predictor = BrainSegmentationPredictor(
            model_path=config['model_path'],
            config=config
        )

        # Get list of input files
        input_dir = config['input_dir']
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                       if f.endswith('.nii') or f.endswith('.nii.gz')]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run prediction
        return predictor.predict_batch(input_files, output_dir)

    if __name__ == "__main__":
        # Example usage
        config = {
            'model_path': 'output/checkpoints/brain_ai_model_best.pth',
            'input_dir': 'data/raw/test',
            'roi_size': [96, 96, 96],
            'sw_batch_size': 4,
            'overlap': 0.5,
            'num_classes': 4  # Background + 3 tissue classes
        }

        output_dir = 'output/predictions'
        results = batch_inference(config, output_dir)

        print(f"Processed {len(results)} images")