import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import monai
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.handlers.stats_handler import StatsHandler
from monai.handlers.tensorboard_handlers import TensorBoardStatsHandler
from monai.inferers import SlidingWindowInferer

# Import local modules
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_metrics, save_metrics_to_csv
from data.dataset import BrainMRIDataset, get_transforms


class BrainAITrainer:
    """Base trainer class for Brain AI models"""

    def __init__(self,
                 config,
                 model,
                 device=None,
                 train_transforms=None,
                 val_transforms=None,
                 optimizer_class=None,
                 loss_function=None,
                 lr_scheduler=None):
        """
        Initialize the trainer with model and training parameters.

        Args:
            config (dict): Configuration dictionary
            model (torch.nn.Module): PyTorch model
            device (torch.device): Device to use for training
            train_transforms (callable): Transforms for training data
            val_transforms (callable): Transforms for validation data
            optimizer_class (torch.optim): Optimizer class
            loss_function (callable): Loss function
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        """
        # Set random seed for reproducibility
        set_determinism(seed=config.get('seed', 42))

        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configuration
        self.config = config
        self.model_name = config.get('model_name', 'brain_ai_model')
        self.num_epochs = config.get('num_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.batch_size = config.get('batch_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.output_dir = Path(config.get('output_dir', './output'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.results_dir = self.output_dir / 'results'

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = model.to(self.device)

        # Transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # Optimizer
        if optimizer_class is None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optimizer_class(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Loss function
        self.loss_function = loss_function if loss_function is not None else nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # Metrics
        self.train_loss_values = []
        self.val_loss_values = []
        self.train_metric_values = []
        self.val_metric_values = []

        # For sliding window inference in case of 3D segmentation
        self.inferer = SlidingWindowInferer(
            roi_size=config.get('roi_size', [64, 64, 64]),
            sw_batch_size=config.get('sw_batch_size', 4),
            overlap=config.get('overlap', 0.5)
        )

        # For dice metric calculation
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

        # Post-processing transforms
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.config.get('num_classes', 2))
        self.post_label = AsDiscrete(to_onehot=self.config.get('num_classes', 2))

        # Best metric value for model saving
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.early_stop_counter = 0

    def prepare_data(self, train_ids, val_ids, test_ids=None):
        """
        Prepare datasets and dataloaders for training and validation.

        Args:
            train_ids (list): List of subject IDs for training
            val_ids (list): List of subject IDs for validation
            test_ids (list, optional): List of subject IDs for testing

        Returns:
            tuple: Train and validation dataloaders
        """
        # Training dataset
        train_dataset = BrainMRIDataset(
            data_dir=self.config['data_dir'],
            clinical_data_path=self.config['clinical_data_path'],
            transform=self.train_transforms,
            subject_ids=train_ids
        )

        # Validation dataset
        val_dataset = BrainMRIDataset(
            data_dir=self.config['data_dir'],
            clinical_data_path=self.config['clinical_data_path'],
            transform=self.val_transforms,
            subject_ids=val_ids
        )

        # Test dataset (optional)
        if test_ids:
            test_dataset = BrainMRIDataset(
                data_dir=self.config['data_dir'],
                clinical_data_path=self.config['clinical_data_path'],
                transform=self.val_transforms,
                subject_ids=test_ids
            )
            self.test_dataset = test_dataset

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available()
        )

        if test_ids:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=torch.cuda.is_available()
            )
            self.test_loader = test_loader

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number

        Returns:
            tuple: Average loss and metrics
        """
        self.model.train()
        epoch_loss = 0
        step = 0

        metric_values = []
        start_time = time.time()

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.loss_function(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate metrics
            if isinstance(outputs, tuple):  # Some models return multiple outputs
                outputs = outputs[0]

            # Convert predictions and labels to one-hot format for dice calculation
            pred = self.post_pred(outputs)
            label = self.post_label(labels)

            # Calculate dice metric
            self.dice_metric(pred, label)
            batch_metric = self.dice_metric_batch(pred, label)
            metric_values.append(batch_metric.mean().item())

            # Update epoch loss
            epoch_loss += loss.item()

            # Print progress
            if step % 10 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Step {step}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Dice: {batch_metric.mean().item():.4f}")

        # Calculate average loss and metrics
        epoch_loss /= step
        mean_dice = np.mean(metric_values)

        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        print(f"Epoch {epoch}/{self.num_epochs} Training completed in {time.time() - start_time:.2f}s")
        print(f"Mean training loss: {epoch_loss:.4f}, Mean training dice: {mean_dice:.4f}")

        return epoch_loss, mean_dice

    def validate_epoch(self, val_loader, epoch):
        """
        Validate the model after one epoch.

        Args:
            val_loader (DataLoader): Validation data loader
            epoch (int): Current epoch number

        Returns:
            tuple: Average validation loss and metrics
        """
        self.model.eval()
        val_loss = 0
        step = 0

        metric_values = []
        start_time = time.time()

        with torch.no_grad():
            for batch_data in val_loader:
                step += 1
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

                # For 3D segmentation, use sliding window inference
                outputs = self.inferer(inputs, self.model)

                # Calculate loss
                loss = self.loss_function(outputs, labels)

                # Calculate metrics
                if isinstance(outputs, tuple):  # Some models return multiple outputs
                    outputs = outputs[0]

                # Convert predictions and labels to one-hot format for dice calculation
                pred = self.post_pred(outputs)
                label = self.post_label(labels)

                # Calculate dice metric
                self.dice_metric(pred, label)
                batch_metric = self.dice_metric_batch(pred, label)
                metric_values.append(batch_metric.mean().item())

                # Update validation loss
                val_loss += loss.item()

        # Calculate average validation loss and metrics
        val_loss /= step
        mean_dice = np.mean(metric_values)

        # Reset metrics
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

        print(f"Epoch {epoch}/{self.num_epochs} Validation completed in {time.time() - start_time:.2f}s")
        print(f"Mean validation loss: {val_loss:.4f}, Mean validation dice: {mean_dice:.4f}")

        return val_loss, mean_dice

    def train(self, train_loader, val_loader):
        """
        Train the model for the specified number of epochs.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Training history
        """
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'best_epoch': -1,
            'best_metric': -float('inf')
        }

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            # Train for one epoch
            train_loss, train_dice = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_dice = self.validate_epoch(val_loader, epoch)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)

            # Learning rate scheduling if applicable
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            # Check if this is the best model so far
            if val_dice > self.best_metric:
                print(f"New best model with validation dice: {val_dice:.4f} (previous: {self.best_metric:.4f})")
                self.best_metric = val_dice
                self.best_epoch = epoch
                history['best_metric'] = val_dice
                history['best_epoch'] = epoch

                # Save best model
                self.save_checkpoint(epoch, is_best=True)

                # Reset early stopping counter
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter}/{self.early_stopping_patience}")

                # Save regular checkpoint
                if epoch % 5 == 0:  # Save every 5 epochs
                    self.save_checkpoint(epoch)

            # Check for early stopping
            if self.early_stop_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

            # Plot training progress
            if epoch % 5 == 0 or epoch == self.num_epochs:
                self.plot_training_progress(history)

        print(f"Training completed. Best model at epoch {self.best_epoch} with validation dice: {self.best_metric:.4f}")

        # Final plots and metrics
        self.plot_training_progress(history, save=True)

        return history

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save a model checkpoint.

        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pth"

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """
        Load a model checkpoint.

        Args:
            checkpoint_path (Path or str): Path to checkpoint
            load_optimizer (bool): Whether to load optimizer state

        Returns:
            int: Epoch number of the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if available
        if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Update best metric
        self.best_metric = checkpoint.get('best_metric', -float('inf'))

        return checkpoint.get('epoch', 0)

    def evaluate(self, test_loader=None):
        """
        Evaluate the model on test data.

        Args:
            test_loader (DataLoader, optional): Test data loader. If None, uses self.test_loader

        Returns:
            dict: Evaluation metrics
        """
        if test_loader is None:
            if not hasattr(self, 'test_loader'):
                raise ValueError("No test_loader provided or found in self")
            test_loader = self.test_loader

        self.model.eval()
        test_metrics = []

        with torch.no_grad():
            for batch_data in test_loader:
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)

                # For 3D segmentation, use sliding window inference
                outputs = self.inferer(inputs, self.model)

                # Calculate metrics
                pred = self.post_pred(outputs)
                label = self.post_label(labels)

                # Calculate dice metric
                self.dice_metric(pred, label)
                batch_metrics = compute_metrics(pred, label)
                test_metrics.append(batch_metrics)

        # Aggregate metrics
        mean_metrics = {}
        for key in test_metrics[0].keys():
            mean_metrics[key] = np.mean([m[key] for m in test_metrics if key in m])

        # Save metrics to CSV
        metrics_path = self.results_dir / f"{self.model_name}_test_metrics.csv"
        save_metrics_to_csv(mean_metrics, metrics_path)

        print(f"Evaluation completed. Results saved to {metrics_path}")
        for key, value in mean_metrics.items():
            print(f"{key}: {value:.4f}")

        return mean_metrics

    def plot_training_progress(self, history, save=False):
        """
        Plot training progress.

        Args:
            history (dict): Training history dictionary
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(15, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot dice
        plt.subplot(1, 2, 2)
        plt.plot(history['train_dice'], label='Training Dice')
        plt.plot(history['val_dice'], label='Validation Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Coefficient')
        plt.title('Training and Validation Dice')
        plt.legend()

        plt.tight_layout()

        if save:
            plt.savefig(self.results_dir / f"{self.model_name}_training_progress.png")

        plt.show()