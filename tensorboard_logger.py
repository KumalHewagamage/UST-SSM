"""TensorBoard Logger for Training"""

import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional


class TensorBoardLogger:
    """Handles all TensorBoard logging operations"""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.accumulated_metrics = {}
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)
        
    def log_training_metrics(self, epoch: int, loss: float, metrics: Dict[str, Any], lr: float):
        """
        Log training metrics for an epoch
        
        Args:
            epoch: Current epoch number
            loss: Training loss
            metrics: Dictionary of additional metrics
            lr: Current learning rate
        """
        self.log_scalar('train/loss', loss, epoch)
        self.log_scalar('train/lr', lr, epoch)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f'train/{key}', value, epoch)
    
    def log_validation_metrics(self, epoch: int, loss: float, metrics: Dict[str, Any]):
        """
        Log validation metrics for an epoch
        
        Args:
            epoch: Current epoch number
            loss: Validation loss
            metrics: Dictionary of additional metrics
        """
        self.log_scalar('val/loss', loss, epoch)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f'val/{key}', value, epoch)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        # Convert all values to strings or numbers for TensorBoard
        hparams_clean = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hparams_clean[k] = v
            else:
                hparams_clean[k] = str(v)
        
        self.writer.add_hparams(hparams_clean, metrics)
    
    def accumulate_metric(self, name: str, value: float):
        """Accumulate a metric value (e.g., for averaging across batches)"""
        if name not in self.accumulated_metrics:
            self.accumulated_metrics[name] = []
        self.accumulated_metrics[name].append(value)
    
    def log_accumulated_metrics(self, step: int, prefix: str = ""):
        """Log accumulated metrics and reset"""
        for name, values in self.accumulated_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                tag = f"{prefix}/{name}" if prefix else name
                self.log_scalar(tag, avg_value, step)
        
        self.accumulated_metrics.clear()
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
