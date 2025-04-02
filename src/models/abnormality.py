"""
Abnormality detection module for brain MRI analysis.
This module provides functionality for detecting abnormalities in brain MRI scans.
"""

import os
import numpy as np
import torch
import nibabel as nib
import logging
from scipy import ndimage
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('abnormality_detection')

# Check if MONAI is available
try:
    import monai
    from monai.networks.nets import DenseNet121
    MONAI_AVAILABLE = True
    logger.info("MONAI is available and will be used for deep learning models")
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning("MONAI is not available. Only statistical methods will be used for abnormality detection.")

class AbnormalityDetector:
    """
    Class for detecting abnormalities in brain MRI scans.
    This class uses a combination of statistical methods and deep learning.
    """

    def __init__(self, model_path=None, device=None, method='hybrid'):
        """
        Initialize the abnormality detector.

        Args:
            model_path (str, optional): Path to a pre-trained model
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
            method (str): Detection method ('statistical', 'deeplearning', or 'hybrid')
        """
        # Metadata
        self.creation_date = "2025-04-02 14:10:54"
        self.created_by = "KishoreKumarKalli"

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Check if method is valid
        if method == 'deeplearning' and not MONAI_AVAILABLE:
            logger.warning("Deep learning method requested but MONAI is not available. Falling back to statistical method.")
            self.method = 'statistical'
        elif method == 'hybrid' and not MONAI_AVAILABLE:
            logger.warning("Hybrid method requested but MONAI is not available. Falling back to statistical method.")
            self.method = 'statistical'
        else:
            self.method = method

        logger.info(f"Initializing abnormality detector with method: {self.method}")
        logger.info(f"Using device: {self.device}")

        # Initialize statistical model
        if self.method in ['statistical', 'hybrid']:
            self.statistical_model = IsolationForest(
                n_estimators=100,
                contamination=0.05,  # Expected proportion of abnormalities
                random_state=42
            )
            logger.info("Statistical model (Isolation Forest) initialized")

        # Initialize deep learning model
        if self.method in ['deeplearning', 'hybrid'] and MONAI_AVAILABLE:
            self.dl_model = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=2  # Normal vs. Abnormal
            ).to(self.device)

            # Load pre-trained model if provided
            if model_path and os.path.exists(model_path):
                try:
                    logger.info(f"Loading pre-trained model from {model_path}")
                    self.dl_model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info("Pre-trained model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading pre-trained model: {str(e)}")
                    logger.warning("Continuing with untrained model")
            else:
                logger.warning("No pre-trained model provided. Deep learning detection may be less accurate.")

            # Set model to evaluation mode
            self.dl_model.eval()
            logger.info("Deep learning model initialized")

    def preprocess_image(self, brain_scan):
        """
        Custom preprocessing function that doesn't rely on MONAI transforms.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        # Add channel dimension if needed
        if len(brain_scan.shape) == 3:
            brain_scan = np.expand_dims(brain_scan, axis=0)

        # Resize if needed (to 128x128x128 which is a common size for 3D models)
        current_shape = brain_scan.shape[1:]
        if current_shape != (128, 128, 128):
            try:
                from scipy.ndimage import zoom
                factors = (1.0,) + tuple(128 / dim for dim in current_shape)
                brain_scan = zoom(brain_scan, factors, order=1)
            except Exception as e:
                logger.warning(f"Resizing failed: {str(e)}. Using original size.")

        # Normalize intensity
        mean = np.mean(brain_scan)
        std = np.std(brain_scan)
        if std > 0:
            brain_scan = (brain_scan - mean) / std

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(brain_scan).float().to(self.device)

        return tensor

    def fit_statistical_model(self, normal_samples):
        """
        Fit the statistical model using normal brain scans.

        Args:
            normal_samples (list): List of normal brain scans (numpy arrays)

        Returns:
            self: The fitted detector
        """
        if self.method not in ['statistical', 'hybrid']:
            logger.warning("Statistical model not used in the current configuration")
            return self

        logger.info(f"Fitting statistical model with {len(normal_samples)} normal samples")

        # Extract features from normal samples
        features = []
        for sample in normal_samples:
            # Extract basic statistical features
            features.append(self._extract_statistical_features(sample))

        # Fit the model
        self.statistical_model.fit(np.array(features))
        logger.info("Statistical model fitting completed")

        return self

    def _extract_statistical_features(self, brain_scan):
        """
        Extract statistical features from a brain scan.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            numpy.ndarray: Feature vector
        """
        # Basic statistical features
        features = [
            np.mean(brain_scan),
            np.std(brain_scan),
            np.percentile(brain_scan, 25),
            np.percentile(brain_scan, 50),
            np.percentile(brain_scan, 75),
            np.max(brain_scan),
            np.min(brain_scan),
            np.sum(brain_scan > 0)  # Non-zero voxel count
        ]

        # Gradient features
        gradient_x = ndimage.sobel(brain_scan, axis=0)
        gradient_y = ndimage.sobel(brain_scan, axis=1)
        gradient_z = ndimage.sobel(brain_scan, axis=2)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)

        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.max(gradient_magnitude)
        ])

        # Regional features (split brain into 8 octants)
        shape = brain_scan.shape
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

        for x_half in [slice(0, center[0]), slice(center[0], None)]:
            for y_half in [slice(0, center[1]), slice(center[1], None)]:
                for z_half in [slice(0, center[2]), slice(center[2], None)]:
                    region = brain_scan[x_half, y_half, z_half]
                    features.append(np.mean(region))

        return np.array(features)

    def detect_with_statistical_model(self, brain_scan):
        """
        Detect abnormalities using the statistical model.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            tuple: (is_abnormal, abnormality_score, abnormality_map)
        """
        # Extract features
        features = self._extract_statistical_features(brain_scan).reshape(1, -1)

        # Get anomaly score (-1 for abnormal, 1 for normal)
        prediction = self.statistical_model.predict(features)
        score = self.statistical_model.decision_function(features)[0]

        # Convert to abnormality score (higher = more abnormal)
        abnormality_score = -score
        is_abnormal = prediction[0] == -1

        # Generate abnormality map
        abnormality_map = self._generate_statistical_abnormality_map(brain_scan)

        logger.info(f"Statistical detection completed at: {self.creation_date}")
        logger.info(f"Detection executed by: {self.created_by}")
        logger.info(f"Abnormality detected: {is_abnormal}, Score: {abnormality_score:.4f}")

        return is_abnormal, abnormality_score, abnormality_map

    def _generate_statistical_abnormality_map(self, brain_scan):
        """
        Generate an abnormality map using statistical methods.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            numpy.ndarray: Abnormality map (higher values indicate potential abnormalities)
        """
        # Create a map of the same size as the input
        abnormality_map = np.zeros_like(brain_scan, dtype=np.float32)

        # Calculate local statistics using a sliding window
        mean_filter = ndimage.uniform_filter(brain_scan, size=5)
        mean_sqr_filter = ndimage.uniform_filter(brain_scan ** 2, size=5)
        variance = mean_sqr_filter - mean_filter ** 2

        # Calculate z-scores (how many standard deviations from the mean)
        # Add small epsilon to avoid division by zero
        std_filter = np.sqrt(variance) + 1e-10
        z_score = (brain_scan - mean_filter) / std_filter

        # Higher absolute z-score indicates potential abnormality
        abnormality_map = np.abs(z_score)

        return abnormality_map

    def detect_with_deep_learning(self, brain_scan):
        """
        Detect abnormalities using the deep learning model.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            tuple: (is_abnormal, abnormality_score, abnormality_map)
        """
        if not MONAI_AVAILABLE:
            logger.error("Deep learning detection requested but MONAI is not available.")
            return False, 0.0, np.zeros_like(brain_scan)

        # Prepare input data using our custom preprocessing
        input_tensor = self.preprocess_image(brain_scan).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = self.dl_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            abnormality_score = probabilities[0, 1].item()  # Probability of abnormal class

        is_abnormal = abnormality_score > 0.5

        # Generate a simple abnormality map (this is a placeholder)
        # In a real implementation, you would use explainability techniques like Grad-CAM
        abnormality_map = np.zeros_like(brain_scan, dtype=np.float32)

        logger.info(f"Deep learning detection completed at: {self.creation_date}")
        logger.info(f"Detection executed by: {self.created_by}")
        logger.info(f"Abnormality detected: {is_abnormal}, Score: {abnormality_score:.4f}")

        return is_abnormal, abnormality_score, abnormality_map

    def detect(self, brain_scan):
        """
        Detect abnormalities in a brain scan using the configured method.

        Args:
            brain_scan (numpy.ndarray): 3D brain scan

        Returns:
            dict: Detection results including:
                 - is_abnormal: Boolean indicating abnormality
                 - abnormality_score: Numerical score (higher = more abnormal)
                 - abnormality_map: 3D map highlighting potential abnormalities
                 - method: Method used for detection
        """
        logger.info(f"Starting abnormality detection with method: {self.method}")

        results = {
            'method': self.method
        }

        if self.method == 'statistical':
            is_abnormal, score, abnormality_map = self.detect_with_statistical_model(brain_scan)
            results.update({
                'is_abnormal': is_abnormal,
                'abnormality_score': score,
                'abnormality_map': abnormality_map
            })

        elif self.method == 'deeplearning' and MONAI_AVAILABLE:
            is_abnormal, score, abnormality_map = self.detect_with_deep_learning(brain_scan)
            results.update({
                'is_abnormal': is_abnormal,
                'abnormality_score': score,
                'abnormality_map': abnormality_map
            })

        elif self.method == 'hybrid' and MONAI_AVAILABLE:
            # Run both methods
            stat_abnormal, stat_score, stat_map = self.detect_with_statistical_model(brain_scan)
            dl_abnormal, dl_score, dl_map = self.detect_with_deep_learning(brain_scan)

            # Combine results (simple averaging for demonstration)
            is_abnormal = stat_abnormal or dl_abnormal
            score = (stat_score + dl_score) / 2

            # Combine abnormality maps (weighted average)
            abnormality_map = 0.4 * stat_map + 0.6 * dl_map

            results.update({
                'is_abnormal': is_abnormal,
                'abnormality_score': score,
                'abnormality_map': abnormality_map,
                'statistical_score': stat_score,
                'deeplearning_score': dl_score
            })
        else:
            # Fallback to statistical method if method is not supported
            logger.warning(f"Method {self.method} not supported or MONAI not available. Using statistical method.")
            is_abnormal, score, abnormality_map = self.detect_with_statistical_model(brain_scan)
            results.update({
                'is_abnormal': is_abnormal,
                'abnormality_score': score,
                'abnormality_map': abnormality_map,
                'method': 'statistical'  # Override the method in results
            })

        logger.info(f"Abnormality detection completed: {results['is_abnormal']}")
        return results

    def save_abnormality_map(self, abnormality_map, original_img, output_path):
        """
        Save abnormality map as a NIfTI file.

        Args:
            abnormality_map (numpy.ndarray): Abnormality map
            original_img (nibabel.nifti1.Nifti1Image): Original NiBabel image object
            output_path (str): Path to save the abnormality map
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.info(f"Saving abnormality map to: {output_path}")

            # Create a new NIfTI image with the abnormality map and original metadata
            abnormality_img = nib.Nifti1Image(abnormality_map, original_img.affine, original_img.header)
            nib.save(abnormality_img, output_path)

            logger.info(f"Abnormality map saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving abnormality map to {output_path}: {str(e)}")
            raise

    def batch_detect(self, input_files, output_dir):
        """
        Perform abnormality detection on a batch of files.

        Args:
            input_files (list): List of dictionaries with 'image' (path) and 'data' (loaded data) keys
            output_dir (str): Directory to save abnormality maps

        Returns:
            dict: Dictionary mapping file paths to detection results
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Starting batch abnormality detection for {len(input_files)} files")

        results = {}
        for i, file_info in enumerate(input_files):
            try:
                # Get file information
                file_path = file_info['image']
                image_data = file_info['data'][0]  # Assuming data is a tuple (image_data, image_obj)
                image_obj = file_info['data'][1]

                # Create output file path
                file_name = os.path.basename(file_path)
                output_name = f"abnormality_{file_name}"
                output_path = os.path.join(output_dir, output_name)

                logger.info(f"Processing file {i + 1}/{len(input_files)}: {file_name}")

                # Perform detection
                detection_result = self.detect(image_data)

                # Save abnormality map
                self.save_abnormality_map(detection_result['abnormality_map'], image_obj, output_path)

                # Store results
                detection_result['output_path'] = output_path
                results[file_path] = detection_result

                logger.info(f"Successfully processed: {file_name} -> {output_name}")
                logger.info(f"Abnormality detected: {detection_result['is_abnormal']}, "
                            f"Score: {detection_result['abnormality_score']:.4f}")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue

        logger.info(f"Batch abnormality detection completed. Processed {len(results)} files.")
        return results