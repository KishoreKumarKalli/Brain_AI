#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brain AI Framework - Main Entry Point

This script serves as the main entry point for the Brain AI framework, which performs
automated brain segmentation, abnormality detection, and statistical analysis in
neuroimaging data. It provides a command-line interface to execute different components
of the framework including data preprocessing, model training, inference, and analysis.

Author: Kishore Kumar Kalligive
Date: 2025-03-31
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import torch
import numpy as np
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import project modules
from src.data.preprocessing import preprocess_data
from src.data.dataset import create_datasets
from src.models.segmentation import BrainSegmentationModel
from src.models.anomaly import AnomalyDetectionModel
from src.training.trainer import ModelTrainer
from src.inference.predict import run_inference
from src.analysis.statistics import run_statistical_analysis
from src.analysis.reporting import generate_report
from src.utils.metrics import evaluate_model


def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"brain_ai_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess(config):
    """Run data preprocessing"""
    logging.info("Starting data preprocessing...")
    preprocess_data(
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
        clinical_dir=config["data"]["clinical_dir"],
        image_size=config["data"]["image_size"],
        spacing=config["data"]["spacing"],
        normalization=config["data"]["intensity_normalization"]
    )
    logging.info("Data preprocessing completed")


def train(config):
    """Train segmentation and anomaly detection models"""
    logging.info("Setting up training...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create datasets and dataloaders
    train_loader, val_loader, test_loader = create_datasets(
        data_dir=config["data"]["processed_dir"],
        clinical_dir=config["data"]["clinical_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["system"]["num_workers"],
        train_val_test_split=config["data"]["train_val_test_split"],
        augmentation=config["data"]["augmentation"],
        pin_memory=config["system"]["pin_memory"]
    )

    # Initialize models
    segmentation_model = BrainSegmentationModel(
        model_type=config["model"]["segmentation"]["type"],
        dimensions=config["model"]["segmentation"]["dimensions"],
        in_channels=config["model"]["segmentation"]["in_channels"],
        out_channels=config["model"]["segmentation"]["out_channels"],
        features=config["model"]["segmentation"]["features"],
        dropout_prob=config["model"]["segmentation"]["dropout"],
        norm=config["model"]["segmentation"]["normalization"],
        act=config["model"]["segmentation"]["activation"],
        pretrained_weights=config["model"]["segmentation"]["pretrained_weights"]
    ).to(device)

    # Initialize anomaly detection model if enabled
    anomaly_model = None
    if config["model"]["anomaly_detection"]["enable"]:
        anomaly_model = AnomalyDetectionModel(
            model_type=config["model"]["anomaly_detection"]["type"],
            dimensions=config["model"]["segmentation"]["dimensions"],
            in_channels=config["model"]["segmentation"]["in_channels"],
            features=config["model"]["anomaly_detection"]["features"],
            attention_levels=config["model"]["anomaly_detection"]["attention_levels"]
        ).to(device)

    # Create trainer and train models
    trainer = ModelTrainer(
        segmentation_model=segmentation_model,
        anomaly_model=anomaly_model,
        device=device,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        scheduler_name=config["training"]["scheduler"],
        loss_seg_name=config["training"]["loss"]["segmentation"],
        loss_anomaly_name=config["training"]["loss"]["anomaly"],
        lambda_seg=config["training"]["loss"]["lambda_seg"],
        lambda_anomaly=config["training"]["loss"]["lambda_anomaly"],
        num_epochs=config["training"]["num_epochs"],
        output_dir=config["paths"]["models_dir"],
        log_dir=config["paths"]["logs_dir"],
        early_stopping=config["training"]["early_stopping"],
        validation_interval=config["training"]["validation"]["interval"],
        metrics=config["training"]["validation"]["metrics"],
        best_metric=config["training"]["validation"]["best_metric"],
        save_best_only=config["training"]["validation"]["save_best_only"],
        amp=config["system"]["amp"]
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Evaluate on test set
    logging.info("Evaluating models on test set...")
    test_metrics = evaluate_model(
        segmentation_model=segmentation_model,
        anomaly_model=anomaly_model,
        test_loader=test_loader,
        device=device,
        metrics=config["training"]["validation"]["metrics"],
        output_dir=os.path.join(config["paths"]["output_dir"], "evaluation")
    )

    logging.info(f"Test metrics: {test_metrics}")


def inference(config, input_path=None, output_path=None):
    """Run inference on input data"""
    logging.info("Setting up inference...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set input and output paths
    if input_path is None:
        input_path = config["data"]["processed_dir"]
    if output_path is None:
        output_path = os.path.join(config["paths"]["output_dir"], "predictions")

    # Load trained models
    segmentation_model = BrainSegmentationModel.load_from_checkpoint(
        os.path.join(config["paths"]["models_dir"], "segmentation_model_best.pth"),
        model_type=config["model"]["segmentation"]["type"],
        dimensions=config["model"]["segmentation"]["dimensions"],
        in_channels=config["model"]["segmentation"]["in_channels"],
        out_channels=config["model"]["segmentation"]["out_channels"],
        features=config["model"]["segmentation"]["features"],
        dropout_prob=config["model"]["segmentation"]["dropout"],
        norm=config["model"]["segmentation"]["normalization"],
        act=config["model"]["segmentation"]["activation"]
    ).to(device)

    anomaly_model = None
    if config["model"]["anomaly_detection"]["enable"]:
        anomaly_model = AnomalyDetectionModel.load_from_checkpoint(
            os.path.join(config["paths"]["models_dir"], "anomaly_model_best.pth"),
            model_type=config["model"]["anomaly_detection"]["type"],
            dimensions=config["model"]["segmentation"]["dimensions"],
            in_channels=config["model"]["segmentation"]["in_channels"],
            features=config["model"]["anomaly_detection"]["features"],
            attention_levels=config["model"]["anomaly_detection"]["attention_levels"]
        ).to(device)

    # Run inference
    results = run_inference(
        segmentation_model=segmentation_model,
        anomaly_model=anomaly_model,
        input_path=input_path,
        output_path=output_path,
        device=device,
        batch_size=config["inference"]["batch_size"],
        sliding_window=config["inference"]["sliding_window"],
        roi_size=config["inference"]["roi_size"],
        overlap=config["inference"]["overlap"],
        sw_batch_size=config["inference"]["sw_batch_size"],
        post_process=config["inference"]["post_process"]
    )

    logging.info(f"Inference completed. Results saved to {output_path}")
    return results


def analyze(config):
    """Run statistical analysis on results"""
    logging.info("Starting statistical analysis...")

    results_dir = os.path.join(config["paths"]["output_dir"], "predictions")
    clinical_dir = config["data"]["clinical_dir"]
    output_dir = os.path.join(config["paths"]["output_dir"], "analysis")

    # Run statistical analysis
    statistics = run_statistical_analysis(
        results_dir=results_dir,
        clinical_dir=clinical_dir,
        clinical_files=config["clinical_data"],
        output_dir=output_dir,
        significance_level=config["analysis"]["significance_level"],
        multiple_comparison_correction=config["analysis"]["multiple_comparison_correction"],
        correlation_method=config["analysis"]["correlation_method"],
        perform_ttest=config["analysis"]["perform_ttest"],
        perform_anova=config["analysis"]["perform_anova"],
        cohort_analysis=config["analysis"]["cohort_analysis"],
        visualization=config["analysis"]["visualization"]
    )

    # Generate report
    report_path = os.path.join(output_dir, "brain_ai_report.pdf")
    generate_report(
        statistics=statistics,
        config=config,
        output_path=report_path
    )

    logging.info(f"Statistical analysis completed. Report saved to {report_path}")


def run_webapp(config):
    """Run the web application"""
    try:
        from webapp.app import create_app

        app = create_app(config)
        app.run(
            host=config["webapp"]["host"],
            port=config["webapp"]["port"],
            debug=config["webapp"]["debug"]
        )
    except ImportError:
        logging.error("Failed to import webapp module. Make sure Flask is installed.")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Brain AI Framework")
    parser.add_argument("--config", type=str, default="./config.yml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "inference", "analyze", "webapp", "all"],
                        default="all", help="Mode to run")
    parser.add_argument("--input", type=str, help="Input data path for inference mode")
    parser.add_argument("--output", type=str, help="Output path for results")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config["paths"]["logs_dir"])

    # Set random seed for reproducibility
    set_random_seed(config["system"]["seed"])

    # Print startup information
    logger.info("=" * 80)
    logger.info(f"Brain AI Framework - Version 1.0.0")
    logger.info(f"Author: Kishore Kumar Kalligive")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {args.mode}")
    logger.info("=" * 80)

    # Create necessary directories
    for dir_path in [
        config["paths"]["processed_data"],
        config["paths"]["models_dir"],
        config["paths"]["logs_dir"],
        config["paths"]["output_dir"]
    ]:
        os.makedirs(dir_path, exist_ok=True)

    # Run the selected mode
    if args.mode == "preprocess" or args.mode == "all":
        preprocess(config)

    if args.mode == "train" or args.mode == "all":
        train(config)

    if args.mode == "inference" or args.mode == "all":
        inference(config, args.input, args.output)

    if args.mode == "analyze" or args.mode == "all":
        analyze(config)

    if args.mode == "webapp":
        run_webapp(config)

    logger.info("Brain AI Framework execution completed successfully")


if __name__ == "__main__":
    main()