#!/usr/bin/env python3
"""
4D Text Grounding Training Script
"""

import os
import sys
import time
import json
import logging
import random
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from datasets.humanml import HumanML3D
from torch.optim.lr_scheduler import MultiStepLR
from models.UST import UST
from models.clip_encoder import create_text_encoder
from modules.tensorboard_logger import TensorBoardLogger
# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

class Config:
    # --- Experiment Settings ---
    experiment_name = "ust_4d_grounding_v6"
    output_dir = "./outputs"
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_freq = 10  # Print training status every N batches

    # --- Data Settings ---
    data_root = 'data/v4.3-wall-humanML3d-2136'
    num_points = 2048
    frames_per_clip = 16
    num_workers = 8  # Adjusted for typical setups; set to 16 if high-end CPU
    
    # --- Model Hyperparameters (UST) ---
    model_params = {
        'radius': 0.7, # 0.3
        'nsamples': 32, # 32
        'spatial_stride': 32,
        'temporal_kernel_size': 3,
        'temporal_stride': 2, # 2
        'dim': 160,
        'heads': 6,
        'mlp_dim': 320,
        'num_classes': 20, # Note: Check if this matches your dataset classes
        'dropout': 0.0,
        'depth': 1, # 1
        'hos_branches_num': 3,
        'encoder_channel': 60
    }

    # --- Text Encoder Settings ---
    text_encoder_type = 'clip'
    text_model_name = "ViT-B/32"

    # --- Training Hyperparameters ---
    epochs = 100
    batch_size = 32
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # --- Scheduler Settings ---
    lr_milestones = [20, 30] # Epochs to step LR
    lr_gamma = 0.1


    # --- Loss Settings ---
    temperature = 0.07

    # --- Validation Settings ---
    k_values = [1, 2, 3, 5, 10] # For Recall@K
    val_every_n = 1  # Validate every N epochs



# ==============================================================================
# 3. UTILITIES & LOGGING
# ==============================================================================

def setup_logger(output_dir):
    """Sets up a simple logger to file and console."""
    log_format = '%(asctime)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_path)


# ==============================================================================
# 4. LOSS FUNCTION
# ==============================================================================

class TextActionAlignment(nn.Module):
    """
    Text-Temporal Alignment Loss using contrastive learning.
    Aligns temporal point cloud features with text descriptions using InfoNCE loss.
    """
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, temporal_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        batch_size = temporal_features.shape[0]
        
        # Ensure float32
        temporal_features = temporal_features.float()
        text_features = text_features.float()
        
        # L2 normalize features
        if self.normalize:
            temporal_features = F.normalize(temporal_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix: (i,j) = similarity between temporal_i and text_j
        logits = temporal_features @ text_features.T / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric contrastive loss
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.T, labels)
        
        return (loss_t2v + loss_v2t) / 2


# ==============================================================================
# 5. VALIDATION EVALUATOR
# ==============================================================================

class ValidationEvaluator:
    """
    Comprehensive validation evaluator for text-to-scene and scene-to-text retrieval.
    """
    def __init__(self, model, text_encoder, criterion, device, k_values=[1, 5, 10]):
        self.model = model
        self.text_encoder = text_encoder
        self.criterion = criterion
        self.device = device
        self.k_values = k_values
    
    def compute_recall_at_k(self, sim_matrix, k, mode='text_to_scene'):
        if mode == 'text_to_scene':
            # For each text, check if correct scene is in top-k
            num = sim_matrix.shape[0]
            if num == 0: return 0.0
            
            # Use topk instead of argsort for efficiency
            _, topk_indices = torch.topk(sim_matrix, k=min(k, num), dim=1, largest=True)
            
            # Check if index i is in the i-th row of topk_indices
            correct = 0
            for i in range(num):
                if i in topk_indices[i]:
                    correct += 1
            return correct / num
        
        elif mode == 'scene_to_text':
            # For each scene, check if correct text is in top-k
            num = sim_matrix.shape[1]
            if num == 0: return 0.0
            
            _, topk_indices = torch.topk(sim_matrix, k=min(k, num), dim=0, largest=True)
            
            correct = 0
            for j in range(num):
                # Check column j
                if j in topk_indices[:, j]:
                    correct += 1
            return correct / num
        return 0.0

    def evaluate(self, dataloader, logger=None, text_projection=None):
        self.model.eval()
        self.text_encoder.eval()
        if text_projection is not None:
            text_projection.eval()
        
        all_scene_embeddings = []
        all_text_embeddings = []
        total_loss = 0.0
        num_batches = 0
        
        # Batch-wise recall tracking
        batch_recalls = {f'recall@{k}_text_to_scene': [] for k in self.k_values}
        batch_recalls.update({f'recall@{k}_scene_to_text': [] for k in self.k_values})

        with torch.no_grad():
            for i, (points, captions) in enumerate(tqdm(dataloader, desc="Validation")):
                points = points.to(self.device)
                
                # Forward pass
                text_features = self.text_encoder(captions)
                temporal_features = self.model(points)
                if text_projection is not None:
                    temporal_features = text_projection(temporal_features)
                
                # Compute Batch Loss
                loss = self.criterion(temporal_features, text_features)
                total_loss += loss.item()
                num_batches += 1

                # Normalize embeddings
                scene_emb = F.normalize(temporal_features, dim=1)
                text_emb = F.normalize(text_features, dim=1)
                
                # Batch-wise similarity and recall
                batch_sim = text_emb @ scene_emb.t()
                for k in self.k_values:
                    batch_recalls[f'recall@{k}_text_to_scene'].append(
                        self.compute_recall_at_k(batch_sim, k, 'text_to_scene')
                    )
                    batch_recalls[f'recall@{k}_scene_to_text'].append(
                        self.compute_recall_at_k(batch_sim, k, 'scene_to_text')
                    )
                
                # Store for global metrics
                all_scene_embeddings.append(scene_emb.cpu())
                all_text_embeddings.append(text_emb.cpu())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Compute average batch-wise recalls
        metrics = {'loss': avg_loss}
        # Use fixed batch size of 32 for evaluation consistency
        eval_batch_size = 32
        for k in self.k_values:
            metrics[f'recall@{k}_text_to_scene_b{eval_batch_size}'] = np.mean(batch_recalls[f'recall@{k}_text_to_scene'])
            metrics[f'recall@{k}_scene_to_text_b{eval_batch_size}'] = np.mean(batch_recalls[f'recall@{k}_scene_to_text'])
        
        # Compute Global Metrics
        if all_scene_embeddings:
            global_scene = torch.cat(all_scene_embeddings, dim=0)
            global_text = torch.cat(all_text_embeddings, dim=0)
            
            # Global Similarity Matrix
            sim = global_text @ global_scene.t()
            
            for k in self.k_values:
                metrics[f'global_recall@{k}_text_to_scene'] = self.compute_recall_at_k(sim, k, 'text_to_scene')
                metrics[f'global_recall@{k}_scene_to_text'] = self.compute_recall_at_k(sim, k, 'scene_to_text')
            
            return metrics
        
        return metrics


# ==============================================================================
# 6. MAIN TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(model, text_encoder, loader, optimizer, criterion, device, epoch, logger, text_projection):
    model.train()
    text_projection.train()
    running_loss = 0.0
    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for i, (points, captions) in enumerate(pbar):
        points = points.to(device)
        
        # Text Encoder (Usually frozen, but requires forward pass)
        with torch.no_grad():
            text_features = text_encoder(captions)

        # Model Forward
        temporal_features = model(points)
        temporal_features = text_projection(temporal_features)

        # Loss
        loss = criterion(temporal_features, text_features)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Update progress bar
        if i % Config.print_freq == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_avg_loss = running_loss / len(loader)
    epoch_time = time.time() - start_time
    
    metrics = {
        'loss': epoch_avg_loss,
        'epoch_time': epoch_time,
        'num_batches': len(loader)
    }
    
    logger.info(f"Train Epoch [{epoch}] Avg Loss: {epoch_avg_loss:.4f}, Time: {epoch_time:.1f}s")
    return metrics


def main():
    # 1. Setup
    exp_dir = os.path.join(Config.output_dir, Config.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = setup_logger(exp_dir)
    logger.info(f"Starting experiment: {Config.experiment_name}")
    logger.info(f"Configuration: {json.dumps(Config.model_params, indent=2)}")
    
    # Initialize TensorBoard logger
    tb_log_dir = os.path.join(exp_dir, 'tensorboard')
    tb_logger = TensorBoardLogger(tb_log_dir)
    logger.info(f"TensorBoard logs: {tb_log_dir}")
    
    set_seed(Config.seed)

    # 2. Data Loading
    logger.info("Initializing Datasets...")
    train_dataset = HumanML3D(
        root=Config.data_root, 
        frames_per_clip=Config.frames_per_clip, 
        num_points=Config.num_points, 
        train=True
    )
    test_dataset = HumanML3D(
        root=Config.data_root, 
        frames_per_clip=Config.frames_per_clip, 
        num_points=Config.num_points, 
        train=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers, 
        pin_memory=True,
        drop_last=True # Important for batch stability
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,  # Fixed batch size for evaluation consistency
        num_workers=Config.num_workers, 
        pin_memory=True
    )

    # 3. Model & Encoder
    logger.info("Building Model...")
    model = UST(**Config.model_params).to(Config.device)
    
    logger.info("Building Text Encoder...")
    text_encoder = create_text_encoder(
        encoder_type=Config.text_encoder_type,
        model_name=Config.text_model_name,
        device=Config.device,
        freeze_encoder=True,
        add_projection=False
    )
    # Ensure text encoder is frozen (no grads) and in eval mode
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    
    text_projection = nn.Linear(483,512).to(Config.device)

    # 4. Optimization
    # Train both the UST model and the projection head; keep text encoder frozen
    params = list(model.parameters()) + list(text_projection.parameters())
    optimizer = torch.optim.SGD(
        params,
        lr=Config.lr,
        momentum=Config.momentum,
        weight_decay=Config.weight_decay
    )

    # Scheduler: use epoch-based MultiStepLR (no warmup)
    lr_scheduler = MultiStepLR(
        optimizer, 
        milestones=Config.lr_milestones, 
        gamma=Config.lr_gamma
    )

    criterion = TextActionAlignment(temperature=Config.temperature).to(Config.device)
    
    # Validator
    evaluator = ValidationEvaluator(
        model=model, 
        text_encoder=text_encoder,
        criterion=criterion, 
        device=Config.device,
        k_values=Config.k_values
    )

    # 5. Training Loop
    best_r1 = 0.0
    start_epoch = 0
    
    # Hyperparameters for logging
    hparams = {
        'lr': Config.lr,
        'batch_size': Config.batch_size,
        'epochs': Config.epochs,
        'temperature': Config.temperature,
        'num_points': Config.num_points,
        'frames_per_clip': Config.frames_per_clip,
        **Config.model_params
    }

    logger.info("Starting Training Loop...")
    for epoch in range(start_epoch, Config.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{Config.epochs}")
        logger.info(f"{'='*50}")
        
        # --- Train ---
        train_metrics = train_one_epoch(
            model, text_encoder, train_loader, optimizer, criterion, Config.device, epoch + 1, logger, text_projection
        )
        
        # Log training metrics to TensorBoard
        current_lr = optimizer.param_groups[0]['lr']
        tb_logger.log_training_metrics(
            epoch=epoch,
            loss=train_metrics['loss'],
            metrics={k: v for k, v in train_metrics.items() if k not in ['loss', 'num_batches']},
            lr=current_lr
        )
        
        # Step scheduler
        lr_scheduler.step()

        # --- Validate ---
        should_validate = (epoch + 1) % Config.val_every_n == 0 or (epoch + 1) == Config.epochs
        
        if should_validate:
            logger.info("Running validation...")
            val_metrics = evaluator.evaluate(test_loader, logger, text_projection)
            
            # Log validation metrics to TensorBoard
            tb_logger.log_validation_metrics(
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics={k: v for k, v in val_metrics.items() if k != 'loss'}
            )
            
            # Log batch-wise Recall@k metrics
            eval_batch_size = 32  # Fixed batch size for evaluation
            logger.info(f"Batch-wise Recall@k (batch_size={eval_batch_size}):")
            for k in Config.k_values:
                t2s = val_metrics.get(f'recall@{k}_text_to_scene_b{eval_batch_size}', 0)
                s2t = val_metrics.get(f'recall@{k}_scene_to_text_b{eval_batch_size}', 0)
                logger.info(f"  R@{k} T→S: {100*t2s:.2f}% | S→T: {100*s2t:.2f}%")
                tb_logger.log_scalar(f'val/recall@{k}_text_to_scene_b{eval_batch_size}', t2s, epoch)
                tb_logger.log_scalar(f'val/recall@{k}_scene_to_text_b{eval_batch_size}', s2t, epoch)
            
            # Log global Recall@k metrics
            logger.info("Global Recall@k:")
            for k in Config.k_values:
                t2s = val_metrics.get(f'global_recall@{k}_text_to_scene', 0)
                s2t = val_metrics.get(f'global_recall@{k}_scene_to_text', 0)
                logger.info(f"  R@{k} T→S: {100*t2s:.2f}% | S→T: {100*s2t:.2f}%")
                tb_logger.log_scalar(f'val/global_recall@{k}_text_to_scene', t2s, epoch)
                tb_logger.log_scalar(f'val/global_recall@{k}_scene_to_text', s2t, epoch)
            
            # Use Recall@1 text-to-scene as best metric
            eval_batch_size = 32  # Fixed batch size for evaluation
            r1_score = val_metrics.get(f'recall@{1}_text_to_scene_b{eval_batch_size}', 0)
            is_best = r1_score > best_r1
            
            if is_best:
                best_r1 = r1_score
                logger.info(f"🎉 New best model! Recall@1 T→S (b{eval_batch_size}): {100*best_r1:.2f}%")
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'text_projection': text_projection.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'best_r1': best_r1,
                'config': Config.model_params
            }, is_best, exp_dir)
            
            if is_best:
                logger.info(f"Best model saved to {exp_dir}")
        
        # Save last checkpoint
        if (epoch + 1) == Config.epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'text_projection': text_projection.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'best_r1': best_r1,
                'config': Config.model_params
            }, False, exp_dir, filename='checkpoint_last.pth.tar')
            logger.info(f"Last model saved to {exp_dir}")

    # Log final hyperparameters
    final_metrics = {'final_best_recall@1': best_r1}
    tb_logger.log_hyperparameters(hparams, final_metrics)
    
    logger.info("Training completed!")
    logger.info(f"Best Recall@1 T→S: {100*best_r1:.2f}%")
    
    # Close TensorBoard
    tb_logger.close()

if __name__ == "__main__":
    main()