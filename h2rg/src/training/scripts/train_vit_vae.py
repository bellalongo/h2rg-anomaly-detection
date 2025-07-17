#!/usr/bin/env python3
"""
Main training script for ViT-VAE anomaly detection
Optimized for fast training at end of internship
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

# Import your modules
from training.utils.vit_vae_trainer import ViTVAETrainer
from models.architectures.temporal_vit_vae import create_temporal_vit_vae
from data_pipeline.loaders.vit_vae_dataset import JobDataLoaderFactory


class FastViTVAETrainer:
    """
    Main trainer orchestrator that sets up and runs the complete training process
    """
    
    def __init__(self, config_path: str, output_dir: str, data_root: str = None):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config()
        
        # Override data root if provided
        if data_root:
            self.config['data']['processed_data_path'] = data_root
        
        # Apply training preset if enabled
        self.apply_training_preset()
        
        # Setup logging
        self.setup_logging()
        
        # Log configuration
        self.log_config()
        
        # Initialize trainer
        self.trainer = ViTVAETrainer(self.config, str(self.output_dir))
            
    def load_config(self) -> dict:
        """Load and validate configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Fix YAML parsing issues with scientific notation
        def fix_numeric_values(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    obj[key] = fix_numeric_values(value)
            elif isinstance(obj, list):
                return [fix_numeric_values(item) for item in obj]
            elif isinstance(obj, str):
                # Try to convert string numbers back to float
                try:
                    if 'e-' in obj or 'e+' in obj:  # Scientific notation
                        return float(obj)
                    elif obj.replace('.', '').replace('-', '').isdigit():  # Regular numbers
                        return float(obj) if '.' in obj else int(obj)
                except ValueError:
                    pass
            return obj
        
        config = fix_numeric_values(config)
        
        # Validate required sections
        required_sections = ['model', 'training', 'data']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def apply_training_preset(self):
        """Apply training preset for different speed/quality tradeoffs"""
        presets = ['fast_training', 'balanced_training', 'quality_training']
        
        for preset_name in presets:
            preset = self.config.get(preset_name, {})
            if preset.get('enabled', False):
                print(f"Applying {preset_name} preset...")
                
                # Update training config with preset values
                for key, value in preset.items():
                    if key != 'enabled':
                        if key in ['max_epochs', 'max_steps_per_epoch', 'batch_size']:
                            self.config['training'][key] = value
                        elif key in ['max_train_samples', 'max_val_samples']:
                            self.config['training'][key] = value
                        elif key == 'num_layers':
                            self.config['model'][key] = value
                
                break
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("=" * 60)
        self.logger.info("ViT-VAE Training for H2RG Anomaly Detection")
        self.logger.info("=" * 60)
    
    def log_config(self):
        """Log the configuration being used"""
        config_file = self.output_dir / 'config_used.yml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_file}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Data path: {self.config['data']['processed_data_path']}")
    
    def setup_model(self):
        """Initialize the ViT-VAE model"""
        self.logger.info("Initializing ViT-VAE model...")
        
        # Create model
        model = create_temporal_vit_vae(self.config['model'])
        
        # Setup model in trainer
        self.trainer.setup_model(model)
        
        return model
    
    def setup_data_loaders(self):
        """Setup data loaders for your job-based data structure"""
        self.logger.info("Setting up data loaders...")
        
        data_config = self.config['data'].copy()
        data_config.update({
            'batch_size': self.config['training']['batch_size'],
            'num_workers': self.config['training'].get('num_workers', 4)
        })
        
        # Add training limits if specified
        for key in ['max_train_samples', 'max_val_samples', 'max_samples_per_job', 'max_val_samples_per_job']:
            if key in self.config['training'] and self.config['training'][key] is not None:
                data_config[key] = self.config['training'][key]
        
        try:
            train_loader, val_loader = JobDataLoaderFactory.create_train_val_loaders(
                job_outputs_dir=self.config['data']['processed_data_path'],
                config=data_config,
                val_split=self.config['data']['val_split']
            )
            
            # Setup in trainer
            self.trainer.setup_data_loaders(train_loader, val_loader)
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Failed to setup data loaders: {e}")
            raise
    
    def setup_optimizer(self):
        """Setup optimizer and training components"""
        self.logger.info("Setting up optimizer...")
        self.trainer.setup_optimizer()
    
    def run_training(self):
        """Run the complete training process"""
        try:
            # Setup all components
            model = self.setup_model()
            train_loader, val_loader = self.setup_data_loaders()
            self.setup_optimizer()
            
            # Print training summary
            self.print_training_summary(train_loader, val_loader)
            
            # Start training
            history = self.trainer.train()
            
            # Training completed successfully
            self.print_completion_summary(history)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def print_training_summary(self, train_loader, val_loader):
        """Print training setup summary"""
        total_steps = self.trainer._calculate_total_steps()
        estimated_time = self.estimate_training_time(total_steps)
        
        self.logger.info("Training Summary:")
        self.logger.info(f"Train samples: {len(train_loader.dataset):,}")
        self.logger.info(f"Val samples: {len(val_loader.dataset):,}")
        self.logger.info(f"Epochs: {self.config['training']['max_epochs']}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Total steps: {total_steps:,}")
        self.logger.info(f"Estimated time: {estimated_time}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.trainer.model.parameters()):,}")
        
        # GPU info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        self.logger.info("=" * 60)
    
    def estimate_training_time(self, total_steps: int) -> str:
        """Estimate training time based on configuration"""
        # Rough estimates based on typical performance
        if self.config['training'].get('use_amp', True):
            seconds_per_step = 0.5  # With mixed precision
        else:
            seconds_per_step = 1.0  # Without mixed precision
        
        # Adjust for gradient checkpointing
        if self.config['model'].get('use_gradient_checkpointing', True):
            seconds_per_step *= 1.2  # ~20% overhead
        
        total_seconds = total_steps * seconds_per_step
        
        if total_seconds < 3600:
            return f"{total_seconds/60:.0f} minutes"
        else:
            return f"{total_seconds/3600:.1f} hours"
    
    def print_completion_summary(self, history: dict):
        """Print training completion summary"""
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        best_epoch = history['val_loss'].index(best_val_loss) + 1 if history['val_loss'] else 0
        
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"Best model: {self.output_dir}/best_model.pth")
        self.logger.info(f"Latest checkpoint: {self.output_dir}/latest_checkpoint.pth")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train ViT-VAE for H2RG anomaly detection")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for models and logs')
    parser.add_argument('--data-root', type=str, 
                       help='Override data root path (your job_outputs directory)')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    parser.add_argument('--test-setup', action='store_true',
                       help='Test setup without training')
    
    args = parser.parse_args()
    
    try:
        # Create output directory with timestamp if not resuming
        if not args.resume:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if not args.output_dir.endswith(timestamp):
                output_dir = Path(args.output_dir) / f"vit_vae_{timestamp}"
            else:
                output_dir = Path(args.output_dir)
        else:
            output_dir = Path(args.output_dir)
        
        # Initialize trainer
        trainer = FastViTVAETrainer(
            config_path=args.config,
            output_dir=str(output_dir),
            data_root=args.data_root
        )
        
        if args.test_setup:
            # Test setup only
            print("Testing setup...")
            trainer.setup_model()
            trainer.setup_data_loaders()
            trainer.setup_optimizer()
            print("Setup test passed!")
            return 0
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint_path = Path(args.resume)
            if checkpoint_path.exists():
                start_epoch = trainer.trainer.load_checkpoint(
                    str(checkpoint_path), 
                    resume_training=True
                )
                trainer.logger.info(f"Resumed training from epoch {start_epoch}")
            else:
                trainer.logger.error(f"Checkpoint not found: {checkpoint_path}")
                return 1
        
        # Start training
        history = trainer.run_training()
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Results saved to: {output_dir}")
        if history['val_loss']:
            best_val_loss = min(history['val_loss'])
            print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Import torch here to avoid issues if not available
    import torch
    
    exit(main())