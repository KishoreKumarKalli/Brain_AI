#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain MRI Analysis Pipeline Runner

This script provides a command line interface to run the complete brain MRI analysis
pipeline, including preprocessing, segmentation, abnormality detection, and volumetric
analysis. It can be used for both single files and batch processing.

Usage:
    python run_pipeline.py --input /path/to/input.nii.gz --output /path/to/output --steps all
    python run_pipeline.py --input /path/to/input_dir --output /path/to/output_dir --batch --steps seg,vol

Author: KishoreKumarKalli
Created: 2025-04-02 15:46:12
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd

# Add parent directory to sys.path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the project
from src.preprocessing import preprocessing
from src.segmentation import segmentation
from src.abnormality import detection
from src.analysis import volumetric
from src.visualization import visualize
from src.utils import file_utils, logging_utils

# Set up logging
logging_utils.setup_logging()
logger = logging.getLogger("brain_ai.pipeline")

# Define constants
VALID_STEPS = ["preproc", "seg", "abn", "vol", "all"]
VALID_FILE_EXTENSIONS = [".nii", ".nii.gz", ".mgz", ".mgh"]


class BrainAnalysisPipeline:
    """
    Main class to run the Brain MRI analysis pipeline.

    This class coordinates the execution of different steps in the pipeline:
    1. Preprocessing (bias correction, skull stripping, normalization)
    2. Segmentation (brain structure segmentation)
    3. Abnormality Detection (detecting anomalies)
    4. Volumetric Analysis (volume measurements of brain structures)
    """

    def __init__(self, input_path, output_dir, steps=None, batch=False,
                 num_workers=None, force_overwrite=False, config_file=None):
        """
        Initialize the pipeline with the given parameters.

        Args:
            input_path (str): Path to input file or directory.
            output_dir (str): Path to output directory.
            steps (list): List of steps to run. Default is ["all"].
            batch (bool): Whether to process multiple files. Default is False.
            num_workers (int): Number of worker processes for batch processing.
            force_overwrite (bool): Whether to overwrite existing files.
            config_file (str): Path to configuration file.
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.steps = steps or ["all"]
        self.batch = batch
        self.force_overwrite = force_overwrite

        # Set up the number of workers
        if num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = num_workers

        # Load configuration
        self.config = self._load_config(config_file)

        # Validate inputs
        self._validate_inputs()

        # Set up output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Initialized pipeline with run ID: {self.run_id}")
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Steps: {', '.join(self.steps)}")
        logger.info(f"Batch mode: {self.batch}")
        logger.info(f"Number of workers: {self.num_workers}")

    def _load_config(self, config_file):
        """
        Load configuration from a JSON file.

        Args:
            config_file (str): Path to configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        default_config = {
            "preprocessing": {
                "bias_correction": True,
                "skull_strip": True,
                "intensity_normalization": True,
                "target_spacing": [1.0, 1.0, 1.0]
            },
            "segmentation": {
                "model_path": "models/segmentation_model.h5",
                "num_classes": 8,
                "patch_size": [128, 128, 128],
                "overlap": 16,
                "batch_size": 2
            },
            "abnormality": {
                "model_path": "models/abnormality_model.h5",
                "threshold": 0.5,
                "patch_size": [64, 64, 64],
                "stride": 32,
                "batch_size": 8
            },
            "volumetric": {
                "reference_csv": "data/reference_volumes.csv",
                "icv_normalization": True
            }
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)

                # Merge with default config
                for section in user_config:
                    if section in default_config:
                        default_config[section].update(user_config[section])
                    else:
                        default_config[section] = user_config[section]

                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Using default configuration")

        return default_config

    def _validate_inputs(self):
        """
        Validate input parameters.

        Raises:
            ValueError: If input parameters are invalid.
        """
        # Check if input path exists
        if not self.input_path.exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")

        # Validate steps
        for step in self.steps:
            if step not in VALID_STEPS:
                raise ValueError(f"Invalid step: {step}. Valid steps: {', '.join(VALID_STEPS)}")

        # If batch mode, check if input is a directory
        if self.batch and not self.input_path.is_dir():
            raise ValueError("In batch mode, input must be a directory")

        # If not batch mode, check if input is a valid file
        if not self.batch:
            if not self.input_path.is_file():
                raise ValueError("Input must be a file in single processing mode")

            if not any(str(self.input_path).lower().endswith(ext) for ext in VALID_FILE_EXTENSIONS):
                raise ValueError(
                    f"Input file has invalid extension. Valid extensions: {', '.join(VALID_FILE_EXTENSIONS)}")

    def run(self):
        """
        Run the brain MRI analysis pipeline.

        Returns:
            dict: Dictionary containing the results and paths to output files.
        """
        start_time = time.time()
        logger.info("Starting Brain MRI Analysis Pipeline")

        if self.batch:
            results = self._run_batch()
        else:
            results = self._run_single(self.input_path)

        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

        # Save pipeline metadata
        self._save_metadata(elapsed_time)

        return results

    def _run_single(self, input_file):
        """
        Run the pipeline on a single input file.

        Args:
            input_file (Path): Path to the input file.

        Returns:
            dict: Dictionary containing the results and paths to output files.
        """
        logger.info(f"Processing file: {input_file}")

        # Create a unique identifier for this file
        file_id = input_file.stem.split('.')[0]

        # Create output subdirectory for this file
        file_output_dir = self.output_dir / file_id
        file_output_dir.mkdir(exist_ok=True)

        # Initialize results dictionary
        results = {
            "input_file": str(input_file),
            "output_dir": str(file_output_dir),
            "steps": {},
            "success": True,
            "error": None
        }

        try:
            # Load the input file
            logger.info(f"Loading input file: {input_file}")
            img = nib.load(str(input_file))

            # Keep track of the current image
            current_img = img

            # Step 1: Preprocessing
            if "all" in self.steps or "preproc" in self.steps:
                logger.info("Running preprocessing step")
                preproc_output_path = file_output_dir / "preprocessed.nii.gz"

                if not preproc_output_path.exists() or self.force_overwrite:
                    preproc_img, preproc_meta = preprocessing.preprocess_brain_mri(
                        img,
                        bias_correction=self.config["preprocessing"]["bias_correction"],
                        skull_strip=self.config["preprocessing"]["skull_strip"],
                        normalize=self.config["preprocessing"]["intensity_normalization"],
                        target_spacing=self.config["preprocessing"]["target_spacing"]
                    )

                    # Save preprocessed image
                    nib.save(preproc_img, str(preproc_output_path))

                    # Save metadata
                    preproc_meta_path = file_output_dir / "preprocessing_metadata.json"
                    with open(preproc_meta_path, 'w') as f:
                        json.dump(preproc_meta, f, indent=2)

                    current_img = preproc_img
                else:
                    logger.info(f"Using existing preprocessed file: {preproc_output_path}")
                    current_img = nib.load(str(preproc_output_path))

                    # Load metadata
                    preproc_meta_path = file_output_dir / "preprocessing_metadata.json"
                    if preproc_meta_path.exists():
                        with open(preproc_meta_path, 'r') as f:
                            preproc_meta = json.load(f)
                    else:
                        preproc_meta = {}

                results["steps"]["preprocessing"] = {
                    "output_file": str(preproc_output_path),
                    "metadata": preproc_meta
                }
            # Step 2: Segmentation
            if "all" in self.steps or "seg" in self.steps:
                logger.info("Running segmentation step")
                seg_output_path = file_output_dir / "segmentation.nii.gz"

                if not seg_output_path.exists() or self.force_overwrite:
                    segmentation_result, seg_meta = segmentation.segment_brain_mri(
                        current_img,
                        model_path=self.config["segmentation"]["model_path"],
                        num_classes=self.config["segmentation"]["num_classes"],
                        patch_size=self.config["segmentation"]["patch_size"],
                        overlap=self.config["segmentation"]["overlap"],
                        batch_size=self.config["segmentation"]["batch_size"]
                    )

                    # Save segmentation result
                    nib.save(segmentation_result, str(seg_output_path))

                    # Save metadata
                    seg_meta_path = file_output_dir / "segmentation_metadata.json"
                    with open(seg_meta_path, 'w') as f:
                        json.dump(seg_meta, f, indent=2)

                    # Generate visualization
                    viz_path = file_output_dir / "segmentation_visualization.png"
                    visualize.create_segmentation_visualization(
                        current_img,
                        segmentation_result,
                        output_path=str(viz_path)
                    )
                else:
                    logger.info(f"Using existing segmentation file: {seg_output_path}")
                    segmentation_result = nib.load(str(seg_output_path))

                    # Load metadata
                    seg_meta_path = file_output_dir / "segmentation_metadata.json"
                    if seg_meta_path.exists():
                        with open(seg_meta_path, 'r') as f:
                            seg_meta = json.load(f)
                    else:
                        seg_meta = {}

                    # Check if visualization exists
                    viz_path = file_output_dir / "segmentation_visualization.png"
                    if not viz_path.exists():
                        visualize.create_segmentation_visualization(
                            current_img,
                            segmentation_result,
                            output_path=str(viz_path)
                        )

                results["steps"]["segmentation"] = {
                    "output_file": str(seg_output_path),
                    "visualization": str(viz_path),
                    "metadata": seg_meta
                }

            # Step 3: Abnormality Detection
            if "all" in self.steps or "abn" in self.steps:
                logger.info("Running abnormality detection step")
                abn_output_path = file_output_dir / "abnormality_map.nii.gz"

                if not abn_output_path.exists() or self.force_overwrite:
                    abnormality_map, abn_score, abn_meta = detection.detect_abnormalities(
                        current_img,
                        model_path=self.config["abnormality"]["model_path"],
                        threshold=self.config["abnormality"]["threshold"],
                        patch_size=self.config["abnormality"]["patch_size"],
                        stride=self.config["abnormality"]["stride"],
                        batch_size=self.config["abnormality"]["batch_size"]
                    )

                    # Save abnormality map
                    nib.save(abnormality_map, str(abn_output_path))

                    # Save metadata
                    abn_meta["abnormality_score"] = float(abn_score)
                    abn_meta_path = file_output_dir / "abnormality_metadata.json"
                    with open(abn_meta_path, 'w') as f:
                        json.dump(abn_meta, f, indent=2)

                    # Generate visualization
                    abn_viz_path = file_output_dir / "abnormality_visualization.png"
                    visualize.create_abnormality_visualization(
                        current_img,
                        abnormality_map,
                        output_path=str(abn_viz_path)
                    )
                else:
                    logger.info(f"Using existing abnormality file: {abn_output_path}")
                    abnormality_map = nib.load(str(abn_output_path))

                    # Load metadata
                    abn_meta_path = file_output_dir / "abnormality_metadata.json"
                    if abn_meta_path.exists():
                        with open(abn_meta_path, 'r') as f:
                            abn_meta = json.load(f)
                            abn_score = abn_meta.get("abnormality_score", 0.0)
                    else:
                        abn_meta = {}
                        abn_score = 0.0

                    # Check if visualization exists
                    abn_viz_path = file_output_dir / "abnormality_visualization.png"
                    if not abn_viz_path.exists():
                        visualize.create_abnormality_visualization(
                            current_img,
                            abnormality_map,
                            output_path=str(abn_viz_path)
                        )

                results["steps"]["abnormality"] = {
                    "output_file": str(abn_output_path),
                    "visualization": str(abn_viz_path),
                    "abnormality_score": abn_score,
                    "metadata": abn_meta
                }

            # Step 4: Volumetric Analysis
            if "all" in self.steps or "vol" in self.steps:
                logger.info("Running volumetric analysis step")
                vol_output_path = file_output_dir / "volumes.csv"

                if not vol_output_path.exists() or self.force_overwrite:
                    # Volumetric analysis requires segmentation
                    if "segmentation" not in results["steps"]:
                        logger.info("Segmentation not available, running it first")
                        seg_output_path = file_output_dir / "segmentation.nii.gz"

                        segmentation_result, seg_meta = segmentation.segment_brain_mri(
                            current_img,
                            model_path=self.config["segmentation"]["model_path"],
                            num_classes=self.config["segmentation"]["num_classes"],
                            patch_size=self.config["segmentation"]["patch_size"],
                            overlap=self.config["segmentation"]["overlap"],
                            batch_size=self.config["segmentation"]["batch_size"]
                        )

                        # Save segmentation result
                        nib.save(segmentation_result, str(seg_output_path))
                    else:
                        seg_output_path = Path(results["steps"]["segmentation"]["output_file"])
                        segmentation_result = nib.load(str(seg_output_path))

                    # Run volumetric analysis
                    volumes_df, vol_meta = volumetric.calculate_volumes(
                        segmentation_result,
                        current_img,
                        reference_csv=self.config["volumetric"]["reference_csv"],
                        icv_normalization=self.config["volumetric"]["icv_normalization"]
                    )

                    # Save volumes data
                    volumes_df.to_csv(vol_output_path, index=False)

                    # Save metadata
                    vol_meta_path = file_output_dir / "volumetric_metadata.json"
                    with open(vol_meta_path, 'w') as f:
                        json.dump(vol_meta, f, indent=2)

                    # Generate visualization
                    vol_viz_path = file_output_dir / "volumes_visualization.png"
                    visualize.create_volume_visualization(
                        volumes_df,
                        output_path=str(vol_viz_path)
                    )
                else:
                    logger.info(f"Using existing volumetric analysis file: {vol_output_path}")
                    volumes_df = pd.read_csv(vol_output_path)

                    # Load metadata
                    vol_meta_path = file_output_dir / "volumetric_metadata.json"
                    if vol_meta_path.exists():
                        with open(vol_meta_path, 'r') as f:
                            vol_meta = json.load(f)
                    else:
                        vol_meta = {}

                    # Check if visualization exists
                    vol_viz_path = file_output_dir / "volumes_visualization.png"
                    if not vol_viz_path.exists():
                        visualize.create_volume_visualization(
                            volumes_df,
                            output_path=str(vol_viz_path)
                        )

                results["steps"]["volumetric"] = {
                    "output_file": str(vol_output_path),
                    "visualization": str(vol_viz_path),
                    "metadata": vol_meta
                }

            # Generate final report
            report_path = file_output_dir / "report.pdf"
            if not report_path.exists() or self.force_overwrite:
                logger.info("Generating final report")
                self._generate_report(results, report_path)

            results["report"] = str(report_path)

        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)

        return results

    def _run_batch(self):
        """
        Run the pipeline on multiple input files in parallel.

        Returns:
            list: List of dictionaries containing results for each file.
        """
        logger.info(f"Running batch processing on {self.input_path}")

        # Collect all valid input files
        input_files = []
        for ext in VALID_FILE_EXTENSIONS:
            input_files.extend(list(self.input_path.glob(f"*{ext}")))

        if not input_files:
            logger.warning(f"No valid input files found in {self.input_path}")
            return []

        logger.info(f"Found {len(input_files)} files to process")

        # Create a batch results metadata file
        batch_meta = {
            "batch_id": self.run_id,
            "num_files": len(input_files),
            "start_time": datetime.now().isoformat(),
            "files": [str(f) for f in input_files]
        }

        batch_meta_path = self.output_dir / "batch_metadata.json"
        with open(batch_meta_path, 'w') as f:
            json.dump(batch_meta, f, indent=2)

        # Process files in parallel
        if self.num_workers > 1:
            logger.info(f"Processing files in parallel with {self.num_workers} workers")
            with mp.Pool(processes=self.num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._run_single, input_files),
                    total=len(input_files),
                    desc="Processing files"
                ))
        else:
            logger.info("Processing files sequentially")
            results = []
            for file in tqdm(input_files, desc="Processing files"):
                results.append(self._run_single(file))

        # Update batch metadata with completion information
        batch_meta["end_time"] = datetime.now().isoformat()
        batch_meta["success_count"] = sum(1 for r in results if r["success"])
        batch_meta["failure_count"] = sum(1 for r in results if not r["success"])

        with open(batch_meta_path, 'w') as f:
            json.dump(batch_meta, f, indent=2)

        # Generate batch summary
        self._generate_batch_summary(results, batch_meta)

        return results

    def _generate_report(self, results, output_path):
        """
        Generate a comprehensive PDF report of the analysis results.

        Args:
            results (dict): The results dictionary.
            output_path (Path): Path to save the report.
        """
        # This is a placeholder for report generation logic
        # In a real implementation, this would use a library like ReportLab,
        # FPDF, or matplotlib to generate a PDF report
        logger.info(f"Generating report to {output_path}")

        # For now, we'll just create a simple text file as a demonstration
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(f"Brain MRI Analysis Report\n")
            f.write(f"-------------------------\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Input file: {results['input_file']}\n\n")

            for step, step_results in results['steps'].items():
                f.write(f"{step.capitalize()} Results:\n")
                f.write(f"  Output file: {step_results['output_file']}\n")
                if "visualization" in step_results:
                    f.write(f"  Visualization: {step_results['visualization']}\n")
                if step == "abnormality" and "abnormality_score" in step_results:
                    f.write(f"  Abnormality Score: {step_results['abnormality_score']}\n")
                f.write("\n")

    def _generate_batch_summary(self, results, batch_meta):
        """
        Generate a summary of batch processing results.

        Args:
            results (list): List of result dictionaries for each file.
            batch_meta (dict): Batch metadata.
        """
        logger.info("Generating batch processing summary")

        summary_path = self.output_dir / "batch_summary.csv"

        # Create a DataFrame to summarize results
        summary_data = []
        for result in results:
            file_summary = {
                "input_file": Path(result["input_file"]).name,
                "success": result["success"],
                "error": result["error"] if not result["success"] else None
            }

            # Add information about which steps were completed
            for step in ["preprocessing", "segmentation", "abnormality", "volumetric"]:
                file_summary[f"{step}_completed"] = step in result.get("steps", {})

            # Add abnormality score if available
            if "abnormality" in result.get("steps", {}):
                file_summary["abnormality_score"] = result["steps"]["abnormality"].get("abnormality_score", None)

            summary_data.append(file_summary)

        # Create and save the DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)

        # Also create an HTML summary for easier viewing
        html_summary_path = self.output_dir / "batch_summary.html"
        summary_df.to_html(html_summary_path, index=False)

        logger.info(f"Batch summary saved to {summary_path}")

    def _save_metadata(self, elapsed_time):
        """
        Save metadata about the pipeline run.

        Args:
            elapsed_time (float): Total execution time in seconds.
        """
        metadata = {
            "run_id": self.run_id,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "steps": self.steps,
            "batch_mode": self.batch,
            "execution_time_seconds": elapsed_time,
            "start_time": (datetime.now() - elapsed_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "version": "1.0.0",
            "author": "KishoreKumarKalli",
            "last_updated": "2025-04-02 15:47:25"
        }

        metadata_path = self.output_dir / "pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Pipeline metadata saved to {metadata_path}")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Brain MRI Analysis Pipeline")

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input file or directory"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output directory"
    )

    parser.add_argument(
        "--steps", "-s",
        default="all",
        help=f"Comma-separated list of steps to run: {', '.join(VALID_STEPS)}"
    )

    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process multiple files in batch mode"
    )

    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes for batch processing (default: CPU count - 1)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite of existing files"
    )

    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger("brain_ai").setLevel(logging.DEBUG)

    # Parse steps
    steps = args.steps.split(",")

    try:
        # Create and run the pipeline
        pipeline = BrainAnalysisPipeline(
            input_path=args.input,
            output_dir=args.output,
            steps=steps,
            batch=args.batch,
            num_workers=args.num_workers,
            force_overwrite=args.force,
            config_file=args.config
        )

        results = pipeline.run()

        # Print summary
        if args.batch:
            success_count = sum(1 for r in results if r["success"])
            failure_count = sum(1 for r in results if not r["success"])
            logger.info(f"Batch processing complete: {success_count} successes, {failure_count} failures")
            logger.info(f"Results saved to {args.output}")
        else:
            if results["success"]:
                logger.info(f"Processing complete. Results saved to {results['output_dir']}")
            else:
                logger.error(f"Processing failed: {results['error']}")

        sys.exit(0 if (not args.batch and results["success"]) or
                      (args.batch and all(r["success"] for r in results))
                 else 1)

    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        sys.exit(1)