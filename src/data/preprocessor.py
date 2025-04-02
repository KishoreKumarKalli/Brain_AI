import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocessor')


class MRIPreprocessor:
    """
    Class for preprocessing T1-weighted MRI scans.
    Includes methods for normalization, skull stripping, and resizing.
    """

    def __init__(self, output_size=(128, 128, 128), normalize=True,
                 skull_strip=False, bias_correct=False):
        """
        Initialize the MRI preprocessor.

        Args:
            output_size (tuple): Target size for resizing (default: (128, 128, 128))
            normalize (bool): Whether to normalize the intensity values (default: True)
            skull_strip (bool): Whether to perform skull stripping (default: False)
            bias_correct (bool): Whether to perform bias field correction (default: False)
        """
        self.output_size = output_size
        self.normalize = normalize
        self.skull_strip = skull_strip
        self.bias_correct = bias_correct
        logger.info(f"Initialized MRIPreprocessor with output_size={output_size}, "
                    f"normalize={normalize}, skull_strip={skull_strip}, "
                    f"bias_correct={bias_correct}")

    def load_nifti(self, file_path):
        """
        Load a NIfTI file and return the image data and header.

        Args:
            file_path (str): Path to the NIfTI file

        Returns:
            tuple: (image_data, image_obj) where image_data is a numpy array and
                  image_obj is the NiBabel image object
        """
        try:
            logger.info(f"Loading NIfTI file: {file_path}")
            img = nib.load(file_path)
            data = img.get_fdata()
            return data, img
        except Exception as e:
            logger.error(f"Error loading NIfTI file {file_path}: {str(e)}")
            raise

    def save_nifti(self, data, original_img, output_path):
        """
        Save preprocessed data as a NIfTI file.

        Args:
            data (numpy.ndarray): Preprocessed image data
            original_img (nibabel.nifti1.Nifti1Image): Original NiBabel image object
            output_path (str): Path to save the preprocessed file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.info(f"Saving preprocessed data to: {output_path}")
            new_img = nib.Nifti1Image(data, original_img.affine, original_img.header)
            nib.save(new_img, output_path)
        except Exception as e:
            logger.error(f"Error saving preprocessed data to {output_path}: {str(e)}")
            raise

        def normalize_intensity(self, data):
            """
            Normalize the intensity values of the image.

            Args:
                data (numpy.ndarray): Image data

            Returns:
                numpy.ndarray: Normalized image data
            """
            logger.info("Normalizing intensity values")
            # Remove outliers (values outside 0.5 and 99.5 percentiles)
            p_low, p_high = np.percentile(data[data > 0], [0.5, 99.5])
            data = np.clip(data, p_low, p_high)

            # Z-score normalization
            if np.std(data) > 0:
                data = (data - np.mean(data)) / np.std(data)
            return data

        def resize_volume(self, data):
            """
            Resize the volume to the target size.

            Args:
                data (numpy.ndarray): Image data

            Returns:
                numpy.ndarray: Resized image data
            """
            logger.info(f"Resizing volume to {self.output_size}")
            return resize(data, self.output_size, order=1, mode='constant',
                          anti_aliasing=True, preserve_range=True)

        def perform_skull_stripping(self, data):
            """
            Perform simple skull stripping based on intensity thresholding.
            This is a simplified approach. For production, consider using dedicated tools.

            Args:
                data (numpy.ndarray): Image data

            Returns:
                numpy.ndarray: Skull-stripped image data
            """
            logger.info("Performing simplified skull stripping")
            # Simple thresholding approach - not as effective as dedicated tools
            # In a real implementation, use tools like FSL BET, FreeSurfer or MONAI's skull stripping

            # Create a brain mask using thresholding
            threshold = np.mean(data)
            brain_mask = data > threshold

            # Apply morphological operations to clean the mask
            from scipy import ndimage
            brain_mask = ndimage.binary_closing(brain_mask, iterations=2)
            brain_mask = ndimage.binary_opening(brain_mask, iterations=2)

            # Apply mask to the original data
            stripped_data = data * brain_mask

            return stripped_data

        def preprocess(self, input_path, output_path=None):
            """
            Preprocess a single MRI scan.

            Args:
                input_path (str): Path to the input NIfTI file
                output_path (str, optional): Path to save the preprocessed file

            Returns:
                numpy.ndarray: Preprocessed image data
            """
            logger.info(f"Preprocessing MRI scan: {input_path}")
            # Load the image
            data, original_img = self.load_nifti(input_path)

            # Skull stripping (if enabled)
            if self.skull_strip:
                data = self.perform_skull_stripping(data)

            # Bias field correction would be here (if enabled)
            # Not implemented in this simplified version

            # Normalize intensity
            if self.normalize:
                data = self.normalize_intensity(data)

            # Resize volume
            if self.output_size:
                data = self.resize_volume(data)

            # Save the preprocessed image if output_path is provided
            if output_path:
                self.save_nifti(data, original_img, output_path)

            return data

        def batch_preprocess(self, input_dir, output_dir, pattern="*.nii.gz"):
            """
            Preprocess all MRI scans in a directory.

            Args:
                input_dir (str): Directory containing input NIfTI files
                output_dir (str): Directory to save preprocessed files
                pattern (str): Glob pattern to match NIfTI files

            Returns:
                list: Paths to preprocessed files
            """
            import glob
            logger.info(f"Batch preprocessing MRI scans in: {input_dir}")
            os.makedirs(output_dir, exist_ok=True)

            # Find all NIfTI files in the input directory
            input_files = glob.glob(os.path.join(input_dir, pattern))
            output_files = []

            for input_file in input_files:
                # Create output file path
                rel_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, rel_path)

                # Preprocess the file
                try:
                    self.preprocess(input_file, output_file)
                    output_files.append(output_file)
                    logger.info(f"Successfully preprocessed: {input_file} -> {output_file}")
                except Exception as e:
                    logger.error(f"Error preprocessing {input_file}: {str(e)}")

            logger.info(f"Batch preprocessing completed. Processed {len(output_files)} files.")
            return output_files

    # Helper functions for clinical data preprocessing
    def preprocess_clinical_data(clinical_data, target_cols=None, impute_method='mean'):
        """
        Preprocess clinical data.

        Args:
            clinical_data (pandas.DataFrame): Clinical data
            target_cols (list, optional): Columns to preprocess
            impute_method (str): Method for imputing missing values ('mean', 'median', 'mode')

        Returns:
            pandas.DataFrame: Preprocessed clinical data
        """
        import pandas as pd
        logger.info("Preprocessing clinical data")

        if target_cols is None:
            # Select only numeric columns
            target_cols = clinical_data.select_dtypes(include=['number']).columns.tolist()

        # Make a copy to avoid modifying the original
        processed_data = clinical_data.copy()

        # Impute missing values
        for col in target_cols:
            if col in processed_data.columns:
                if processed_data[col].isna().any():
                    if impute_method == 'mean':
                        val = processed_data[col].mean()
                    elif impute_method == 'median':
                        val = processed_data[col].median()
                    elif impute_method == 'mode':
                        val = processed_data[col].mode()[0]
                    else:
                        raise ValueError(f"Invalid impute_method: {impute_method}")

                    processed_data[col].fillna(val, inplace=True)
                    logger.info(f"Imputed missing values in column '{col}' using {impute_method}")

        return processed_data

    def normalize_clinical_data(clinical_data, target_cols=None):
        """
        Normalize clinical data using standardization (z-score).

        Args:
            clinical_data (pandas.DataFrame): Clinical data
            target_cols (list, optional): Columns to normalize

        Returns:
            pandas.DataFrame: Normalized clinical data
        """
        logger.info("Normalizing clinical data")

        if target_cols is None:
            # Select only numeric columns
            target_cols = clinical_data.select_dtypes(include=['number']).columns.tolist()

        # Make a copy to avoid modifying the original
        normalized_data = clinical_data.copy()

        # Apply standardization
        scaler = StandardScaler()
        normalized_data[target_cols] = scaler.fit_transform(normalized_data[target_cols])

        return normalized_data