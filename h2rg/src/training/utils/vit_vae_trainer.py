"""
ViT-VAE Trainer class for H2RG anomaly detection
Optimized for fast training with gradient checkpointing and mixed precision
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# For optional wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ViTVAETrainer:
    """
    Fast trainer for ViT-VAE with optimizations for quick training
    Includes gradient checkpointing, mixed precision, and efficient checkpointing
    """
    
    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup device and distributed training
        self.setup_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_kl_loss': [],
            'val_kl_loss': [],
            'train_temporal_loss': [],
            'val_temporal_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Initialize wandb if available and enabled
        if WANDB_AVAILABLE and config.get('use_wandb', False):
            self.init_wandb()
        else:
            self.use_wandb = False
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / 'training.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 60)
        self.logger.info("ViT-VAE Trainer Initialized")
        self.logger.info("=" * 60)
    
    def setup_device(self):
        """Setup computing device and optimizations"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Optimize CUDA settings for performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Memory management
            torch.cuda.empty_cache()
            
        else:
            self.device = torch.device('cpu')
            self.logger.warning("CUDA not available, using CPU (training will be slow)")
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.get('wandb_project', 'h2rg-vit-vae'),
                config=self.config,
                name=f"vit_vae_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dir=str(self.output_dir)
            )
            self.use_wandb = True
            self.logger.info("Wandb initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def setup_model(self, model: nn.Module):
        """Setup model for training"""
        self.model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model setup complete:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Log gradient checkpointing status
        if hasattr(self.model, 'use_gradient_checkpointing'):
            self.logger.info(f"  Gradient checkpointing: {'Success' if self.model.use_gradient_checkpointing else 'Failure'}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Optimizer with optimized settings for fast convergence
        # DEBUG: Check what we're actually getting
        lr_value = self.config['training']['learning_rate']
        print(f"DEBUG: Learning rate value: {lr_value}")
        print(f"DEBUG: Learning rate type: {type(lr_value)}")
        print(f"DEBUG: Weight decay value: {self.config['training']['weight_decay']}")
        print(f"DEBUG: Weight decay type: {type(self.config['training']['weight_decay'])}")
        
        # Convert to float just in case
        lr_value = float(lr_value)
        wd_value = float(self.config['training']['weight_decay'])
        
        # Optimizer with optimized settings for fast convergence
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr_value,  # Use the converted value
            weight_decay=wd_value,  # Use the converted value
            betas=(0.9, 0.999),
            eps=1e-8
        )
        # self.optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.config['training']['learning_rate'],
        #     weight_decay=self.config['training']['weight_decay'],
        #     betas=(0.9, 0.999),
        #     eps=1e-8
        # )
        
        # Learning rate scheduler for fast training
        total_steps = self._calculate_total_steps()
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1000
        )
        
        # Mixed precision scaler
        if self.config['training'].get('use_amp', True):
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        self.logger.info(f"Optimizer setup: AdamW with OneCycleLR ({total_steps} total steps)")
    
    def _calculate_total_steps(self) -> int:
        """Calculate total training steps"""
        steps_per_epoch = len(self.train_loader)
        if self.config['training'].get('max_steps_per_epoch'):
            steps_per_epoch = min(steps_per_epoch, self.config['training']['max_steps_per_epoch'])
        
        total_steps = self.config['training']['max_epochs'] * steps_per_epoch
        return total_steps
    
    def setup_data_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Setup data loaders"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.logger.info(f"Data loaders setup:")
        self.logger.info(f"  Train batches: {len(self.train_loader)}")
        self.logger.info(f"  Validation batches: {len(self.val_loader)}")
        self.logger.info(f"  Batch size: {train_loader.batch_size}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        # Metrics tracking
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        temporal_loss_sum = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['max_epochs']}",
            leave=False
        )
        
        max_steps = self.config['training'].get('max_steps_per_epoch')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            patches = batch['patch'].to(self.device, non_blocking=True)
            temporal_labels = batch['temporal_label'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(patches)
                    loss_dict = self.model.compute_loss(
                        patches, outputs, temporal_labels,
                        beta=self.config['training'].get('kl_weight', 0.1),
                        temporal_weight=self.config['training'].get('temporal_weight', 0.2)
                    )
                
                # Backward pass with mixed precision
                self.scaler.scale(loss_dict['total_loss']).backward()
                
                # Gradient clipping
                max_grad_norm = self.config['training'].get('max_grad_norm')
                if max_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(patches)
                loss_dict = self.model.compute_loss(
                    patches, outputs, temporal_labels,
                    beta=self.config['training'].get('kl_weight', 0.1),
                    temporal_weight=self.config['training'].get('temporal_weight', 0.2)
                )
                
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                max_grad_norm = self.config['training'].get('max_grad_norm')
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            recon_loss_sum += loss_dict['recon_loss'].item()
            kl_loss_sum += loss_dict['kl_loss'].item()
            if 'temporal_loss' in loss_dict:
                temporal_loss_sum += loss_dict['temporal_loss'].item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.get('log_every_n_steps', 50) == 0:
                wandb.log({
                    'train/batch_loss': loss_dict['total_loss'].item(),
                    'train/batch_recon_loss': loss_dict['recon_loss'].item(),
                    'train/batch_kl_loss': loss_dict['kl_loss'].item(),
                    'train/learning_rate': current_lr,
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
            
            # Early stopping for fast training
            if max_steps and batch_idx >= max_steps:
                self.logger.info(f"Stopping epoch early at step {batch_idx}")
                break
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches,
            'temporal_loss': temporal_loss_sum / num_batches if temporal_loss_sum > 0 else 0.0
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        temporal_loss_sum = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                patches = batch['patch'].to(self.device, non_blocking=True)
                temporal_labels = batch['temporal_label'].to(self.device, non_blocking=True)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(patches)
                        loss_dict = self.model.compute_loss(
                            patches, outputs, temporal_labels,
                            beta=self.config['training'].get('kl_weight', 0.1),
                            temporal_weight=self.config['training'].get('temporal_weight', 0.2)
                        )
                else:
                    outputs = self.model(patches)
                    loss_dict = self.model.compute_loss(
                        patches, outputs, temporal_labels,
                        beta=self.config['training'].get('kl_weight', 0.1),
                        temporal_weight=self.config['training'].get('temporal_weight', 0.2)
                    )
                
                total_loss += loss_dict['total_loss'].item()
                recon_loss_sum += loss_dict['recon_loss'].item()
                kl_loss_sum += loss_dict['kl_loss'].item()
                if 'temporal_loss' in loss_dict:
                    temporal_loss_sum += loss_dict['temporal_loss'].item()
                num_batches += 1
        
        val_metrics = {
            'total_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches,
            'temporal_loss': temporal_loss_sum / num_batches if temporal_loss_sum > 0 else 0.0
        }
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/total_loss': val_metrics['total_loss'],
                'val/recon_loss': val_metrics['recon_loss'],
                'val/kl_loss': val_metrics['kl_loss'],
                'epoch': epoch
            })
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        
        # Checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_metrics['total_loss']
            self.best_epoch = epoch
            self.logger.info(f"New best model saved (val_loss: {val_metrics['total_loss']:.4f})")
        
        # Save periodic checkpoints
        save_every = self.config['training'].get('save_every_n_epochs', 10)
        if epoch % save_every == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
        
        # Save lightweight model state dict
        if is_best:
            model_path = self.output_dir / 'best_model.pth'
            torch.save(self.model.state_dict(), model_path)
    
    def update_history(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Update training history"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_metrics['total_loss'])
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['train_recon_loss'].append(train_metrics['recon_loss'])
        self.history['val_recon_loss'].append(val_metrics['recon_loss'])
        self.history['train_kl_loss'].append(train_metrics['kl_loss'])
        self.history['val_kl_loss'].append(val_metrics['kl_loss'])
        self.history['train_temporal_loss'].append(train_metrics['temporal_loss'])
        self.history['val_temporal_loss'].append(val_metrics['temporal_loss'])
        self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
        
        # Save history to file
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self) -> Dict:
        """Main training loop"""
        self.logger.info("Starting ViT-VAE training...")
        
        # Training parameters
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training'].get('patience', 10)
        patience_counter = 0
        
        start_time = time.time()
        
        try:
            for epoch in range(1, max_epochs + 1):
                epoch_start = time.time()
                
                # Training
                train_metrics = self.train_epoch(epoch)
                
                # Validation
                val_metrics = self.validate(epoch)
                
                # Update history
                self.update_history(epoch, train_metrics, val_metrics)
                
                epoch_time = time.time() - epoch_start
                
                # Check for improvement
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Logging
                self.logger.info(
                    f"Epoch {epoch:3d}/{max_epochs} ({epoch_time:.1f}s) - "
                    f"Train: {train_metrics['total_loss']:.4f} "
                    f"(recon: {train_metrics['recon_loss']:.4f}, kl: {train_metrics['kl_loss']:.4f}) | "
                    f"Val: {val_metrics['total_loss']:.4f} "
                    f"(recon: {val_metrics['recon_loss']:.4f}, kl: {val_metrics['kl_loss']:.4f}) "
                    f"{'Best' if is_best else ''}"
                )
                
                # Save checkpoint
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best)
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs (patience: {patience})")
                    break
                
                # Memory cleanup
                torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        # Training summary
        total_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed!")
        self.logger.info(f"Total time: {total_time/3600:.1f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info("=" * 60)
        
        # Save final model
        final_model_path = self.output_dir / 'final_model.pth'
        torch.save(self.model.state_dict(), final_model_path)
        
        # Close wandb if used
        if self.use_wandb:
            wandb.finish()
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True) -> int:
        """
        Load checkpoint and optionally resume training state
        Returns: epoch number to resume from
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Resume training state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # Restore training history
            self.history = checkpoint.get('history', self.history)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            
            epoch = checkpoint['epoch']
            self.logger.info(f"Resumed training state from epoch {epoch}")
            return epoch + 1
        else:
            self.logger.info("Loaded model weights only")
            return 0