#!/usr/bin/env python3
"""
ViT-VAE Inference and Evaluation Script
Run inference on your trained model and analyze anomaly detection performance
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)
import h5py

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from models.architectures.temporal_vit_vae import create_temporal_vit_vae
from data_pipeline.loaders.vit_vae_dataset import JobDataLoaderFactory


class ViTVAEInference:
    """
    Inference and evaluation class for trained ViT-VAE models
    """
    
    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        self.logger.info(f"ViT-VAE inference initialized on {self.device}")
    
    def _load_config(self) -> Dict:
        """Load model configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model architecture
        model = create_temporal_vit_vae(self.config['model'])
        
        # Load weights
        if str(self.model_path).endswith('.pth'):
            if 'checkpoint' in str(self.model_path):
                # Load from checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Load state dict directly
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            raise ValueError(f"Unsupported model file format: {self.model_path}")
        
        model = model.to(self.device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Loaded model with {total_params:,} parameters")
        
        return model
    
    def predict_batch(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run inference on a batch of patches"""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            patches = patches.to(self.device)
            
            # Forward pass
            outputs = self.model(patches, return_components=True)
            
            # Compute reconstruction error (anomaly score)
            recon_error = F.mse_loss(outputs['reconstructed'], patches, reduction='none')
            anomaly_scores = torch.mean(recon_error.view(recon_error.shape[0], -1), dim=1)
            
            # Get temporal classification predictions
            temporal_probs = F.softmax(outputs['temporal_classification'], dim=1)
            temporal_preds = torch.argmax(temporal_probs, dim=1)
            
            return {
                'reconstructed': outputs['reconstructed'].cpu(),
                'latent': outputs['latent'].cpu(),
                'anomaly_scores': anomaly_scores.cpu(),
                'temporal_predictions': temporal_preds.cpu(),
                'temporal_probabilities': temporal_probs.cpu(),
                'mu': outputs['mu'].cpu(),
                'logvar': outputs['logvar'].cpu()
            }
    
    def evaluate_dataset(self, data_loader, output_dir: str = None) -> Dict:
        """Evaluate model on a dataset"""
        self.logger.info("Running evaluation on dataset...")
        
        all_anomaly_scores = []
        all_recon_errors = []
        all_temporal_preds = []
        all_temporal_labels = []
        all_latents = []
        all_metadata = []
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            patches = batch['patch']
            temporal_labels = batch['temporal_label']
            
            # Run inference
            predictions = self.predict_batch(patches)
            
            # Compute losses
            with torch.no_grad():
                patches_gpu = patches.to(self.device)
                temporal_labels_gpu = temporal_labels.to(self.device)
                
                outputs = self.model(patches_gpu)
                loss_dict = self.model.compute_loss(
                    patches_gpu, outputs, temporal_labels_gpu,
                    beta=self.config['training'].get('kl_weight', 0.1),
                    temporal_weight=self.config['training'].get('temporal_weight', 0.2)
                )
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
            
            # Store results
            all_anomaly_scores.extend(predictions['anomaly_scores'].numpy())
            all_temporal_preds.extend(predictions['temporal_predictions'].numpy())
            all_temporal_labels.extend(temporal_labels.numpy())
            all_latents.append(predictions['latent'].numpy())
            
            # Compute reconstruction errors per sample
            recon_errors = F.mse_loss(
                predictions['reconstructed'], patches, reduction='none'
            ).view(patches.shape[0], -1).mean(dim=1)
            all_recon_errors.extend(recon_errors.numpy())
            
            # Store metadata
            for i in range(len(patches)):
                all_metadata.append({
                    'job_folder': batch['job_folder'][i],
                    'base_name': batch['base_name'][i],
                    'anomaly_score': float(predictions['anomaly_scores'][i]),
                    'temporal_pred': int(predictions['temporal_predictions'][i]),
                    'temporal_label': int(temporal_labels[i])
                })
        
        # Combine all latents
        all_latents = np.vstack(all_latents)
        
        # Convert to numpy arrays
        all_anomaly_scores = np.array(all_anomaly_scores)
        all_temporal_preds = np.array(all_temporal_preds)
        all_temporal_labels = np.array(all_temporal_labels)
        all_recon_errors = np.array(all_recon_errors)
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        
        # Temporal classification metrics
        temporal_accuracy = (all_temporal_preds == all_temporal_labels).mean()
        
        # Anomaly detection metrics (using reconstruction error)
        # Consider anything with temporal label > 0 as anomaly
        anomaly_labels = (all_temporal_labels > 0).astype(int)
        
        # ROC-AUC for anomaly detection
        if len(np.unique(anomaly_labels)) > 1:
            roc_auc = roc_auc_score(anomaly_labels, all_recon_errors)
            pr_auc = average_precision_score(anomaly_labels, all_recon_errors)
        else:
            roc_auc = 0.0
            pr_auc = 0.0
        
        # Classification report
        class_names = ['normal', 'snowball', 'cosmic_ray', 'telegraph', 'hot_pixel']
        available_classes = np.unique(all_temporal_labels)
        available_class_names = [class_names[i] for i in available_classes if i < len(class_names)]
        
        classification_rep = classification_report(
            all_temporal_labels, all_temporal_preds,
            target_names=available_class_names,
            labels=available_classes,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'avg_loss': avg_loss,
            'temporal_accuracy': temporal_accuracy,
            'anomaly_roc_auc': roc_auc,
            'anomaly_pr_auc': pr_auc,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(all_temporal_labels, all_temporal_preds).tolist(),
            'num_samples': len(all_anomaly_scores),
            'class_distribution': {
                class_names[i]: int(np.sum(all_temporal_labels == i)) 
                for i in range(len(class_names)) 
                if i in all_temporal_labels
            }
        }
        
        # Save detailed results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save numerical results
            with open(output_path / 'evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save arrays for further analysis
            np.savez(
                output_path / 'evaluation_data.npz',
                anomaly_scores=all_anomaly_scores,
                temporal_predictions=all_temporal_preds,
                temporal_labels=all_temporal_labels,
                reconstruction_errors=all_recon_errors,
                latent_representations=all_latents
            )
            
            # Save metadata
            with open(output_path / 'sample_metadata.json', 'w') as f:
                json.dump(all_metadata, f, indent=2)
            
            # Create visualizations
            self._create_evaluation_plots(
                all_temporal_labels, all_temporal_preds, 
                all_recon_errors, anomaly_labels,
                all_latents, output_path
            )
        
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"  Average Loss: {avg_loss:.4f}")
        self.logger.info(f"  Temporal Accuracy: {temporal_accuracy:.4f}")
        self.logger.info(f"  Anomaly ROC-AUC: {roc_auc:.4f}")
        self.logger.info(f"  Anomaly PR-AUC: {pr_auc:.4f}")
        
        return results
    
    def _create_evaluation_plots(self, temporal_labels: np.ndarray, 
                               temporal_preds: np.ndarray,
                               recon_errors: np.ndarray,
                               anomaly_labels: np.ndarray,
                               latents: np.ndarray,
                               output_path: Path):
        """Create evaluation visualizations"""
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(temporal_labels, temporal_preds)
        class_names = ['normal', 'snowball', 'cosmic_ray', 'telegraph', 'hot_pixel']
        available_classes = np.unique(temporal_labels)
        available_class_names = [class_names[i] for i in available_classes if i < len(class_names)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=available_class_names,
                   yticklabels=available_class_names)
        plt.title('Temporal Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve for Anomaly Detection
        if len(np.unique(anomaly_labels)) > 1:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(anomaly_labels, recon_errors)
            roc_auc = roc_auc_score(anomaly_labels, recon_errors)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Anomaly Detection ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(anomaly_labels, recon_errors)
            pr_auc = average_precision_score(anomaly_labels, recon_errors)
            
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Anomaly Detection Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / 'pr_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Reconstruction Error Distribution
        plt.figure(figsize=(12, 8))
        
        # Split by anomaly type
        for i, class_name in enumerate(['normal', 'snowball', 'cosmic_ray', 'telegraph', 'hot_pixel']):
            mask = temporal_labels == i
            if np.any(mask):
                plt.hist(recon_errors[mask], bins=50, alpha=0.7, 
                        label=f'{class_name} (n={np.sum(mask)})', density=True)
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution by Anomaly Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'reconstruction_error_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Latent Space Visualization (first 2 dimensions)
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        class_names = ['normal', 'snowball', 'cosmic_ray', 'telegraph', 'hot_pixel']
        
        for i in range(len(class_names)):
            mask = temporal_labels == i
            if np.any(mask):
                plt.scatter(latents[mask, 0], latents[mask, 1], 
                          c=colors[i], label=f'{class_names[i]} (n={np.sum(mask)})',
                          alpha=0.6, s=20)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization (First 2 Dimensions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'latent_space_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Class-wise performance metrics
        class_names = ['normal', 'snowball', 'cosmic_ray', 'telegraph', 'hot_pixel']
        available_classes = np.unique(temporal_labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy per class
        class_accuracies = []
        class_counts = []
        for i in available_classes:
            mask = temporal_labels == i
            if np.any(mask):
                accuracy = np.mean(temporal_preds[mask] == i)
                class_accuracies.append(accuracy)
                class_counts.append(np.sum(mask))
            else:
                class_accuracies.append(0)
                class_counts.append(0)
        
        axes[0, 0].bar(range(len(available_classes)), class_accuracies)
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Per-Class Accuracy')
        axes[0, 0].set_xticks(range(len(available_classes)))
        axes[0, 0].set_xticklabels([class_names[i] for i in available_classes], rotation=45)
        
        # Class distribution
        axes[0, 1].bar(range(len(available_classes)), class_counts)
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xticks(range(len(available_classes)))
        axes[0, 1].set_xticklabels([class_names[i] for i in available_classes], rotation=45)
        
        # Reconstruction error by class
        for i, class_name in enumerate(class_names):
            mask = temporal_labels == i
            if np.any(mask):
                axes[1, 0].boxplot(recon_errors[mask], positions=[i], widths=0.6)
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Reconstruction Error')
        axes[1, 0].set_title('Reconstruction Error by Class')
        axes[1, 0].set_xticks(range(len(class_names)))
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        
        # Latent space variance by class
        class_variances = []
        for i in range(len(class_names)):
            mask = temporal_labels == i
            if np.any(mask):
                variance = np.var(latents[mask], axis=0).mean()
                class_variances.append(variance)
            else:
                class_variances.append(0)
        
        axes[1, 1].bar(range(len(class_names)), class_variances)
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Mean Latent Variance')
        axes[1, 1].set_title('Latent Space Variance by Class')
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'class_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ViT-VAE Inference and Evaluation")
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model file (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to data directory (job_outputs)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--job-folders', nargs='+', 
                       help='Specific job folders to evaluate (default: all)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inferencer = ViTVAEInference(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
        
        # Setup data loader
        data_config = {
            'patch_size': inferencer.config['data']['patch_size'],
            'min_anomaly_score': 0.0,  # Include all samples for evaluation
            'batch_size': args.batch_size,
            'num_workers': 4,
            'temporal_classification': True,
            'normalize_patches': inferencer.config['data'].get('normalize_patches', True)
        }
        
        # Add max samples for testing
        if args.max_samples:
            data_config['max_samples_per_job'] = args.max_samples // 10  # Distribute across jobs
        
        if args.job_folders:
            # Create loader for specific job folders
            data_loader = JobDataLoaderFactory.create_inference_loader(
                job_outputs_dir=args.data_root,
                config=data_config,
                job_folders=args.job_folders
            )
        else:
            # Create loader for all available data
            data_loader = JobDataLoaderFactory.create_inference_loader(
                job_outputs_dir=args.data_root,
                config=data_config
            )
        
        print(f"Evaluating {len(data_loader.dataset)} samples...")
        
        # Run evaluation
        results = inferencer.evaluate_dataset(
            data_loader=data_loader,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("Evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Temporal Accuracy: {results['temporal_accuracy']:.3f}")
        print(f"Anomaly ROC-AUC: {results['anomaly_roc_auc']:.3f}")
        print(f"Anomaly PR-AUC: {results['anomaly_pr_auc']:.3f}")
        print(f"Evaluated {results['num_samples']} samples")
        print("\nClass Distribution:")
        for class_name, count in results['class_distribution'].items():
            print(f"  {class_name}: {count}")
        print("="*60)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())