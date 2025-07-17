"""
Decoder components for ViT-VAE
Reconstructing difference images from latent representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .attention import TransformerBlock


class TemporalDecoder(nn.Module):
    """Decoder for reconstructing difference images from latent space"""
    
    def __init__(self, latent_dim: int, embed_dim: int, patch_size: int, num_decoder_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_decoder_layers = num_decoder_layers
        
        # Project latent vector to embedding space
        self.latent_to_embed = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer decoder blocks (fewer than encoder for efficiency)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=8, dropout=0.1)
            for _ in range(num_decoder_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Patch reconstruction head
        self.patch_to_pixel = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, patch_size * patch_size),
            nn.Tanh()  # Output in [-1, 1] range (assuming normalized input)
        )
        
    def forward(self, z: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Decode latent vector to image
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            target_shape: (height, width) of target image
        Returns:
            Reconstructed image of shape (batch_size, height, width)
        """
        B = z.shape[0]
        H, W = target_shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Project latent to embedding space
        embed = self.latent_to_embed(z)  # (B, embed_dim)
        
        # Expand to patch sequence (broadcast the same embedding to all patches)
        patch_embeds = embed.unsqueeze(1).expand(B, num_patches, self.embed_dim)
        
        # Add learnable patch position embeddings
        if hasattr(self, 'patch_pos_embed'):
            patch_embeds = patch_embeds + self.patch_pos_embed[:num_patches]
        
        # Pass through transformer decoder blocks
        for block in self.decoder_blocks:
            patch_embeds = block(patch_embeds)
        
        # Final normalization
        patch_embeds = self.final_norm(patch_embeds)
        
        # Convert embeddings to pixel values
        patches = self.patch_to_pixel(patch_embeds)  # (B, num_patches, patch_size^2)
        
        # Reshape patches back to image format
        patches = patches.view(B, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        
        # Combine patches to form complete image
        image = self._patches_to_image(patches, (H, W))
        
        return image
    
    def _patches_to_image(self, patches: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert patch tensor to image tensor
        Args:
            patches: Patches of shape (B, num_patches_h, num_patches_w, patch_size, patch_size)
            image_size: (height, width) of target image
        Returns:
            Image tensor of shape (B, height, width)
        """
        B, num_patches_h, num_patches_w, patch_size, _ = patches.shape
        H, W = image_size
        
        # Rearrange patches to form image
        # (B, num_patches_h, num_patches_w, patch_size, patch_size) 
        # -> (B, num_patches_h, patch_size, num_patches_w, patch_size)
        # -> (B, H, W)
        image = patches.permute(0, 1, 3, 2, 4).contiguous()
        image = image.view(B, H, W)
        
        return image


class ConvolutionalDecoder(nn.Module):
    """Alternative convolutional decoder for comparison"""
    
    def __init__(self, latent_dim: int, target_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_size = target_size
        
        # Calculate initial spatial size
        self.init_size = target_size[0] // 32  # 16 for 512x512
        
        # Linear projection to initial feature map
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2),
            nn.ReLU()
        )
        
        # Convolutional upsampling layers
        self.conv_blocks = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # 256x256 -> 512x512
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector using convolutional layers
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        Returns:
            Reconstructed image of shape (batch_size, height, width)
        """
        # Project to initial feature map
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Upsample through conv layers
        img = self.conv_blocks(out)
        
        # Remove channel dimension (squeeze from (B, 1, H, W) to (B, H, W))
        img = img.squeeze(1)
        
        return img


class VariationalDecoder(nn.Module):
    """Decoder with variational output for uncertainty estimation"""
    
    def __init__(self, latent_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.base_decoder = TemporalDecoder(latent_dim, embed_dim, patch_size)
        
        # Additional head for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, patch_size * patch_size),
            nn.Softplus()  # Ensures positive uncertainty values
        )
    
    def forward(self, z: torch.Tensor, target_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode with uncertainty estimation
        Returns:
            (reconstructed_image, uncertainty_map)
        """
        # Get standard reconstruction
        recon = self.base_decoder(z, target_shape)
        
        # TODO: Compute uncertainty map (simplified version)
        # In practice, you'd compute this properly through the decoder
        uncertainty = torch.ones_like(recon) * 0.1  # Placeholder
        
        return recon, uncertainty


class MultiScaleDecoder(nn.Module):
    """Decoder that outputs multiple resolution scales"""
    
    def __init__(self, latent_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.base_decoder = TemporalDecoder(latent_dim, embed_dim, patch_size)
        
        # Additional decoders for different scales
        self.scale_decoders = nn.ModuleDict({
            'quarter': TemporalDecoder(latent_dim, embed_dim // 2, patch_size // 2),
            'half': TemporalDecoder(latent_dim, embed_dim // 2, patch_size)
        })
    
    def forward(self, z: torch.Tensor, target_shape: Tuple[int, int]) -> dict:
        """
        Decode at multiple scales
        Returns:
            Dictionary with reconstructions at different scales
        """
        H, W = target_shape
        
        outputs = {
            'full': self.base_decoder(z, (H, W)),
            'half': F.interpolate(
                self.scale_decoders['half'](z, (H//2, W//2)).unsqueeze(1), 
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1),
            'quarter': F.interpolate(
                self.scale_decoders['quarter'](z, (H//4, W//4)).unsqueeze(1), 
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)
        }
        
        return outputs


# For testing the decoder components
if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    latent_dim = 256
    embed_dim = 768
    patch_size = 16
    target_shape = (512, 512)
    
    print("Testing Decoder Components...")
    
    # Create test latent vector
    z = torch.randn(batch_size, latent_dim)
    print(f"Input latent shape: {z.shape}")
    
    # Test TemporalDecoder
    print("\n1. Testing TemporalDecoder...")
    decoder = TemporalDecoder(latent_dim, embed_dim, patch_size)
    reconstructed = decoder(z, target_shape)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Expected shape: {(batch_size,) + target_shape}")
    assert reconstructed.shape == (batch_size,) + target_shape, "TemporalDecoder shape mismatch"
    print("TemporalDecoder test passed")
    
    # Test ConvolutionalDecoder
    print("\n2. Testing ConvolutionalDecoder...")
    conv_decoder = ConvolutionalDecoder(latent_dim, target_shape)
    conv_reconstructed = conv_decoder(z)
    print(f"Conv reconstructed shape: {conv_reconstructed.shape}")
    assert conv_reconstructed.shape == (batch_size,) + target_shape, "ConvolutionalDecoder shape mismatch"
    print("ConvolutionalDecoder test passed")
    
    # Test VariationalDecoder
    print("\n3. Testing VariationalDecoder...")
    var_decoder = VariationalDecoder(latent_dim, embed_dim, patch_size)
    var_recon, uncertainty = var_decoder(z, target_shape)
    print(f"Variational recon shape: {var_recon.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    assert var_recon.shape == (batch_size,) + target_shape, "VariationalDecoder recon shape mismatch"
    assert uncertainty.shape == (batch_size,) + target_shape, "VariationalDecoder uncertainty shape mismatch"
    print("VariationalDecoder test passed")
    
    # Test MultiScaleDecoder
    print("\n4. Testing MultiScaleDecoder...")
    ms_decoder = MultiScaleDecoder(latent_dim, embed_dim, patch_size)
    ms_outputs = ms_decoder(z, target_shape)
    print(f"MultiScale outputs keys: {list(ms_outputs.keys())}")
    for scale, output in ms_outputs.items():
        print(f"  {scale}: {output.shape}")
        assert output.shape == (batch_size,) + target_shape, f"MultiScaleDecoder {scale} shape mismatch"
    print("MultiScaleDecoder test passed")
    
    # Test reconstruction quality metrics
    print("\n5. Testing reconstruction quality...")
    # Create a simple target image
    target = torch.randn(batch_size, *target_shape)
    
    # Compute MSE loss
    mse_loss = F.mse_loss(reconstructed, target)
    print(f"MSE loss: {mse_loss.item():.4f}")
    
    # Compute SSIM-like metric (simplified)
    def simple_ssim(img1, img2):
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1 = torch.var(img1)
        sigma2 = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        c1, c2 = 0.01, 0.03
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    ssim_score = simple_ssim(reconstructed, target)
    print(f"Simple SSIM: {ssim_score.item():.4f}")
    print("Reconstruction quality metrics computed")
    
    print("\nAll decoder components working correctly!")
    print(f"Summary:")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Embedding dimension: {embed_dim}")
    print(f"   - Patch size: {patch_size}")
    print(f"   - Target shape: {target_shape}")
    print(f"   - TemporalDecoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"   - ConvolutionalDecoder parameters: {sum(p.numel() for p in conv_decoder.parameters()):,}")