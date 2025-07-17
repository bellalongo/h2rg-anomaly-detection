"""
Main ViT-VAE model for temporal anomaly detection in H2RG difference images
Combines all components for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Optional

from .layers.embedding import PatchEmbedding, PositionalEncoding, CLSToken, PatchProcessor
from .layers.attention import TransformerBlock
from .layers.decoder import TemporalDecoder


class TemporalViTVAE(nn.Module):
    """
    Vision Transformer VAE designed for temporal difference image anomaly detection
    Optimized for H2RG detector anomalies with gradient checkpointing
    """
    
    def __init__(self, 
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 6,
                 latent_dim: int = 256,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True,
                 image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        
        # Store configuration
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.image_size = image_size
        
        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        # Patch embedding components
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, self.num_patches + 1)
        self.cls_token = CLSToken(embed_dim)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Pre-VAE layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # VAE encoding heads
        self.mu_head = nn.Linear(embed_dim, latent_dim)
        self.logvar_head = nn.Linear(embed_dim, latent_dim)
        
        # Temporal anomaly classification head
        self.temporal_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5)  # 5 classes: normal, snowball, cosmic_ray, telegraph, hot_pixel
        )
        
        # Decoder
        self.decoder = TemporalDecoder(latent_dim, embed_dim, patch_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            nn.init.normal_(m, std=0.02)
    
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from difference images
        Args:
            x: Input difference images of shape (batch_size, height, width)
        Returns:
            Patches of shape (batch_size, num_patches, patch_size^2)
        """
        return PatchProcessor.extract_patches(x, self.patch_size)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space
        Args:
            x: Input difference images of shape (batch_size, height, width)
        Returns:
            (mu, logvar) - VAE encoding parameters
        """
        B = x.shape[0]
        
        # Extract and embed patches
        patches = self.extract_patches(x)  # (B, num_patches, patch_size^2)
        patch_embeds = self.patch_embedding(patches)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        sequence = self.cls_token(patch_embeds)  # (B, num_patches + 1, embed_dim)
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence)
        
        # Process through transformer blocks with optional gradient checkpointing
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                sequence = checkpoint(block, sequence, use_reentrant=False)
            else:
                sequence = block(sequence)
        
        # Use CLS token for global representation
        cls_output = self.norm(sequence[:, 0])  # (B, embed_dim)
        
        # VAE encoding
        mu = self.mu_head(cls_output)
        logvar = self.logvar_head(cls_output)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Decode latent vector to image
        Args:
            z: Latent vector
            target_shape: Target image shape (default: self.image_size)
        Returns:
            Reconstructed image
        """
        if target_shape is None:
            target_shape = self.image_size
        return self.decoder(z, target_shape)
    
    def forward(self, x: torch.Tensor, return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal difference image
        Args:
            x: Input difference image (B, H, W)
            return_components: Whether to return intermediate components
        Returns:
            Dictionary containing model outputs
        """
        # Encoding
        mu, logvar = self.encode(x)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Temporal classification
        temporal_pred = self.temporal_classifier(z)
        
        # Decoding
        reconstructed = self.decode(z, x.shape[-2:])
        
        outputs = {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'latent': z,
            'temporal_classification': temporal_pred
        }
        
        if return_components:
            # Add intermediate representations for analysis
            patches = self.extract_patches(x)
            outputs.update({
                'patches': patches,
                'temporal_probabilities': F.softmax(temporal_pred, dim=1)
            })
        
        return outputs
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                     temporal_labels: Optional[torch.Tensor] = None,
                     beta: float = 0.1, temporal_weight: float = 0.2,
                     reduction: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with optional temporal classification loss
        Args:
            x: Original input images
            outputs: Model outputs from forward pass
            temporal_labels: Ground truth temporal labels (optional)
            beta: Weight for KL divergence (beta-VAE)
            temporal_weight: Weight for temporal classification loss
            reduction: Loss reduction method ('mean', 'sum', 'none')
        Returns:
            Dictionary of loss components
        """
        batch_size = x.shape[0]
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(outputs['reconstructed'], x, reduction=reduction)
        
        # KL divergence loss
        mu = outputs['mu']
        logvar = outputs['logvar']
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if reduction == 'mean':
            kl_loss = kl_loss.mean()
        elif reduction == 'sum':
            kl_loss = kl_loss.sum()
        # else keep as per-sample for reduction='none'
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kl_loss
        
        # Initialize with VAE loss
        total_loss = vae_loss
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'vae_loss': vae_loss
        }
        
        # Add temporal classification loss if labels provided
        if temporal_labels is not None:
            temporal_loss = F.cross_entropy(
                outputs['temporal_classification'], 
                temporal_labels, 
                reduction=reduction
            )
            total_loss = vae_loss + temporal_weight * temporal_loss
            
            loss_dict.update({
                'total_loss': total_loss,
                'temporal_loss': temporal_loss
            })
        
        return loss_dict
    
    def encode_dataset(self, dataloader, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode entire dataset to latent space for analysis
        Args:
            dataloader: DataLoader for the dataset
            device: Device to run inference on
        Returns:
            (latent_vectors, labels) - concatenated results
        """
        self.eval()
        latents = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['patch'].to(device)
                temporal_labels = batch.get('temporal_label', None)
                
                # Encode to latent space
                mu, _ = self.encode(x)
                latents.append(mu.cpu())
                
                if temporal_labels is not None:
                    labels.append(temporal_labels.cpu())
        
        latents = torch.cat(latents, dim=0)
        labels = torch.cat(labels, dim=0) if labels else None
        
        return latents, labels
    
    def get_anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error
        Args:
            x: Input images
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            recon_error = F.mse_loss(
                outputs['reconstructed'], x, reduction='none'
            )
            # Average over spatial dimensions
            anomaly_scores = torch.mean(recon_error.view(recon_error.shape[0], -1), dim=1)
        
        return anomaly_scores
    
    def interpolate_latent(self, z1: torch.Tensor, z2: torch.Tensor, 
                          steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two latent vectors
        Args:
            z1, z2: Latent vectors to interpolate between
            steps: Number of interpolation steps
        Returns:
            Interpolated images
        """
        self.eval()
        alphas = torch.linspace(0, 1, steps).to(z1.device)
        interpolated_images = []
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = self.decode(z_interp.unsqueeze(0))
                interpolated_images.append(img)
        
        return torch.cat(interpolated_images, dim=0)


def create_temporal_vit_vae(config: Dict) -> TemporalViTVAE:
    """
    Factory function to create ViT-VAE model from configuration
    Args:
        config: Dictionary containing model configuration
    Returns:
        Initialized TemporalViTVAE model
    """
    return TemporalViTVAE(
        patch_size=config.get('patch_size', 16),
        embed_dim=config.get('embed_dim', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 6),
        latent_dim=config.get('latent_dim', 256),
        dropout=config.get('dropout', 0.1),
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', True),
        image_size=config.get('image_size', (512, 512))
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by component"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    component_params = {}
    for name, module in model.named_children():
        component_params[name] = sum(p.numel() for p in module.parameters())
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'components': component_params
    }


# For testing the complete model
if __name__ == "__main__":
    # Test configuration
    config = {
        'patch_size': 16,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 6,
        'latent_dim': 256,
        'dropout': 0.1,
        'use_gradient_checkpointing': True,
        'image_size': (512, 512)
    }
    
    print("Testing Complete ViT-VAE Model...")
    
    # Create model
    model = create_temporal_vit_vae(config)
    print(f"Model created successfully")
    
    # Count parameters
    param_count = count_parameters(model)
    print(f"Model parameters:")
    print(f"   Total: {param_count['total']:,}")
    print(f"   Trainable: {param_count['trainable']:,}")
    for comp, count in param_count['components'].items():
        print(f"   {comp}: {count:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 512, 512)
    temporal_labels = torch.randint(0, 5, (batch_size,))
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    outputs = model(x, return_components=True)
    
    print(f"Forward pass successful!")
    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape}")
    
    # Test loss computation
    print(f"\nTesting loss computation...")
    loss_dict = model.compute_loss(x, outputs, temporal_labels)
    
    print(f"Loss computation successful!")
    print(f"Losses:")
    for key, value in loss_dict.items():
        print(f"   {key}: {value.item():.4f}")
    
    # Test anomaly scoring
    print(f"\nTesting anomaly scoring...")
    anomaly_scores = model.get_anomaly_scores(x)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Anomaly scores: {anomaly_scores}")
    print(f"Anomaly scoring successful!")
    
    # Test latent interpolation
    print(f"\nTesting latent interpolation...")
    with torch.no_grad():
        mu1, _ = model.encode(x[:1])
        mu2, _ = model.encode(x[1:2])
        interpolated = model.interpolate_latent(mu1[0], mu2[0], steps=5)
    print(f"Interpolated images shape: {interpolated.shape}")
    print(f"Latent interpolation successful!")
    
    print(f"\nComplete ViT-VAE model working correctly!")
    print(f"Model summary:")
    print(f"   - Input size: {config['image_size']}")
    print(f"   - Patch size: {config['patch_size']}")
    print(f"   - Number of patches: {model.num_patches}")
    print(f"   - Embedding dimension: {config['embed_dim']}")
    print(f"   - Latent dimension: {config['latent_dim']}")
    print(f"   - Transformer layers: {config['num_layers']}")
    print(f"   - Total parameters: {param_count['total']:,}")
    print(f"   - Gradient checkpointing: {'Success' if config['use_gradient_checkpointing'] else 'Failure'}")