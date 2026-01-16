#!/usr/bin/env python3
"""
MSR Action3D Training Script (Classification)
"""

import os
import sys
import glob
import time
import json
import logging
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.msr_point import MSRAction3D
# Import your model
# Ensure 'models' folder is in your python path
try:
    from models import UST
except ImportError:
    print("WARNING: Could not import 'models'. Ensure your project structure is correct.")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    # --- Experiment Settings ---
    experiment_name = "ust_msr_action_v3"
    output_dir = "./outputs"
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 10

    # --- Data Settings ---
    data_root = 'data/msr/pcds' 
    num_points = 2048
    frames_per_clip = 24 
    num_workers = 8
    num_classes = 20  
    step_between_clips = 4    

    # --- Model Hyperparameters (UST Backbone) ---
    model_params = {
        'radius': 0.3, 
        'nsamples': 32,
        'spatial_stride': 32,
        'temporal_kernel_size': 3,
        'temporal_stride': 2,
        'dim': 160,
        'heads': 6,
        'mlp_dim': 320,
        'num_classes': 20, 
        'dropout': 0.5,     
        'depth': 1,
        'hos_branches_num': 3,
        'encoder_channel': 60
    }
    
    # --- Classifier Head Settings ---
    feature_dim = 483  # The specific output dimension of your UST token
    head_hidden_dim = 256
    head_dropout = 0.5

    # --- Training Hyperparameters ---
    epochs = 100
    batch_size = 16       
    lr = 0.01 
    momentum = 0.9
    weight_decay = 1e-4
    
    # --- Scheduler Settings ---
    lr_milestones = [40, 70]
    lr_gamma = 0.1

# ==============================================================================
# 2. DATASET CLASS
# ==============================================================================

# ==============================================================================
# 3. MODEL WRAPPER (Backbone + MLP Head)
# ==============================================================================

class ActionClassifier(nn.Module):
    """
    Wraps the UST backbone with an MLP classification head.
    Input: Point Cloud Clip [B, T, N, 3]
    Output: Class Logits [B, num_classes]
    """
    def __init__(self, backbone, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        
        # MLP Prediction Head
        # Projects 483 -> 256 -> 20
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [B, T, N, 3]
        
        # Get token from backbone
        # Expected shape: [B, 483]
        features = self.backbone(x)
        
        # Pass through MLP
        logits = self.head(features)
        
        return logits

# ==============================================================================
# 4. UTILITIES
# ==============================================================================

def setup_logger(output_dir):
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(os.path.join(output_dir, 'train.log')),
        logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_plot(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

# ==============================================================================
# 5. TRAINING & VALIDATION LOOPS
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)
        
        # Forward
        outputs = model(points)  
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Validating", leave=False):
            points, labels = points.to(device), labels.to(device)
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    # Setup
    exp_dir = os.path.join(Config.output_dir, Config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_dir)
    set_seed(Config.seed)
    
    logger.info(f"Experiment: {Config.experiment_name}")
    logger.info(f"Config: {json.dumps(Config.model_params, indent=2)}")

    # Data
    train_ds = MSRAction3D(Config.data_root, Config.frames_per_clip, Config.num_points, split='train')
    test_ds = MSRAction3D(Config.data_root, Config.frames_per_clip, Config.num_points, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, 
                              num_workers=Config.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, 
                             num_workers=Config.num_workers, pin_memory=True)

    # Model Construction
    logger.info("Initializing UST Backbone and Classifier Head...")
    backbone = UST.UST(**Config.model_params)
    
    # Wrap backbone with our new MLP Head
    model = ActionClassifier(
        backbone=backbone,
        input_dim=Config.feature_dim,
        hidden_dim=Config.head_hidden_dim,
        num_classes=Config.num_classes,
        dropout=Config.head_dropout
    ).to(Config.device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimizer now sees parameters from both backbone and the new head
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.lr, momentum=Config.momentum, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=Config.lr_milestones, gamma=Config.lr_gamma)

    # Training Loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(Config.epochs):
        logger.info(f"Epoch {epoch+1}/{Config.epochs}")
        
        # Train
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, Config.device)
        
        # Validate
        v_loss, v_acc = validate(model, test_loader, criterion, Config.device)
        
        # Scheduler
        scheduler.step()
        
        # Logging
        logger.info(f"  Train -> Loss: {t_loss:.4f} | Acc: {t_acc:.2f}%")
        logger.info(f"  Val   -> Loss: {v_loss:.4f} | Acc: {v_acc:.2f}%")
        
        # Update History
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        # Checkpoint
        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f"  >>> New Best Model Saved! ({best_acc:.2f}%)")
            
        # Plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_plot(history, exp_dir)

    # Final Save
    torch.save(model.state_dict(), os.path.join(exp_dir, 'last_model.pth'))
    save_plot(history, exp_dir)
    logger.info(f"Training Complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()