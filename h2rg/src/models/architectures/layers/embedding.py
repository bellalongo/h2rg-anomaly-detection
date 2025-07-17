"""
Embedding layers for ViT-VAE
Patch embedding and positional encoding for H2RG difference images
"""

import torch
import torch.nn as nn
import math
import numpy as np


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings for transformer processing"""
    
    def __init__(self, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Convert patches to embeddings
        Args:
            x: Patches tensor of shape (batch_size, num_patches, patch_size * patch_size)
        Returns:
            Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        # Project flattened patches to embedding dimension
        x = self.projection(x)  # (B, num_patches, embed_dim)
        x = self.norm(x)        # Layer normalization for stable training
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for patch positions"""
    
    def __init__(self, embed_dim: int, max_patches: int = 4096):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_patches, embed_dim)
        position = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1)
        
        # Create wavelengths for different dimensions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_patches, embed_dim)
        
    def forward(self, x):
        """
        Add positional encoding to embeddings
        Args:
            x: Embedded patches of shape (batch_size, seq_len, embed_dim)
        Returns:
            Position-encoded embeddings of same shape
        """
        # Add positional encoding (broadcasting across batch dimension)
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding as an alternative to sinusoidal"""
    
    def __init__(self, embed_dim: int, max_patches: int = 4096):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        """Add learnable positional encoding"""
        return x + self.pos_embedding[:, :x.size(1)]


class PatchProcessor:
    """Utility class for extracting patches from difference images"""
    
    @staticmethod
    def extract_patches(images: torch.Tensor, patch_size: int, stride: int = None) -> torch.Tensor:
        """
        Extract patches from images
        Args:
            images: Input images of shape (batch_size, height, width)
            patch_size: Size of each patch
            stride: Stride for patch extraction (default: patch_size for non-overlapping)
        Returns:
            Patches of shape (batch_size, num_patches, patch_size * patch_size)
        """
        if stride is None:
            stride = patch_size
            
        B, H, W = images.shape
        
        # Ensure images are divisible by patch size
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image size ({H}, {W}) must be divisible by patch_size {patch_size}"
        
        # Extract patches using unfold
        patches = images.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        # patches shape: (B, num_patches_h, num_patches_w, patch_size, patch_size)
        
        # Reshape to (B, num_patches, patch_size^2)
        num_patches_h = patches.shape[1]
        num_patches_w = patches.shape[2]
        patches = patches.contiguous().view(B, num_patches_h * num_patches_w, -1)
        
        return patches
    
    @staticmethod
    def reconstruct_from_patches(patches: torch.Tensor, image_size: tuple, patch_size: int) -> torch.Tensor:
        """
        Reconstruct images from patches
        Args:
            patches: Patches of shape (batch_size, num_patches, patch_size * patch_size)
            image_size: (height, width) of original image
            patch_size: Size of each patch
        Returns:
            Reconstructed images of shape (batch_size, height, width)
        """
        B, num_patches, patch_dim = patches.shape
        H, W = image_size
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Reshape patches to include spatial dimensions
        patches = patches.view(B, num_patches_h, num_patches_w, patch_size, patch_size)
        
        # Combine patches to form images
        images = patches.permute(0, 1, 3, 2, 4).contiguous()
        images = images.view(B, H, W)
        
        return images


class CLSToken(nn.Module):
    """Learnable CLS token for global representation"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, x):
        """
        Prepend CLS token to sequence
        Args:
            x: Embedded patches of shape (batch_size, num_patches, embed_dim)
        Returns:
            Sequence with CLS token of shape (batch_size, num_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)


# For testing the embedding components
if __name__ == "__main__":
    # Test parameters
    batch_size = 2
    image_size = (512, 512)
    patch_size = 16
    embed_dim = 768
    
    print("Testing Embedding Components...")
    
    # Create test image
    images = torch.randn(batch_size, *image_size)
    print(f"Input images shape: {images.shape}")
    
    # Test patch extraction
    print("\n1. Testing patch extraction...")
    patches = PatchProcessor.extract_patches(images, patch_size)
    num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
    expected_shape = (batch_size, num_patches, patch_size * patch_size)
    print(f"Extracted patches shape: {patches.shape}")
    print(f"Expected shape: {expected_shape}")
    assert patches.shape == expected_shape, "Patch extraction shape mismatch"
    print("Patch extraction test passed")
    
    # Test patch embedding
    print("\n2. Testing patch embedding...")
    patch_embed = PatchEmbedding(patch_size, embed_dim)
    embedded_patches = patch_embed(patches)
    expected_embed_shape = (batch_size, num_patches, embed_dim)
    print(f"Embedded patches shape: {embedded_patches.shape}")
    print(f"Expected shape: {expected_embed_shape}")
    assert embedded_patches.shape == expected_embed_shape, "Patch embedding shape mismatch"
    print("Patch embedding test passed")
    
    # Test positional encoding
    print("\n3. Testing positional encoding...")
    pos_enc = PositionalEncoding(embed_dim, max_patches=num_patches + 1)
    pos_encoded = pos_enc(embedded_patches)
    print(f"Position-encoded shape: {pos_encoded.shape}")
    assert pos_encoded.shape == embedded_patches.shape, "Positional encoding shape mismatch"
    print("Positional encoding test passed")
    
    # Test CLS token
    print("\n4. Testing CLS token...")
    cls_token = CLSToken(embed_dim)
    with_cls = cls_token(pos_encoded)
    expected_cls_shape = (batch_size, num_patches + 1, embed_dim)
    print(f"With CLS token shape: {with_cls.shape}")
    print(f"Expected shape: {expected_cls_shape}")
    assert with_cls.shape == expected_cls_shape, "CLS token shape mismatch"
    print("CLS token test passed")
    
    # Test patch reconstruction
    print("\n5. Testing patch reconstruction...")
    reconstructed = PatchProcessor.reconstruct_from_patches(patches, image_size, patch_size)
    print(f"Reconstructed images shape: {reconstructed.shape}")
    print(f"Expected shape: {images.shape}")
    assert reconstructed.shape == images.shape, "Reconstruction shape mismatch"
    
    # Check reconstruction accuracy (should be exact for non-overlapping patches)
    reconstruction_error = torch.mean((images - reconstructed) ** 2)
    print(f"Reconstruction error: {reconstruction_error.item():.2e}")
    assert reconstruction_error < 1e-10, "Reconstruction not exact"
    print("Patch reconstruction test passed")
    
    print("\nAll embedding components working correctly!")
    print(f"Summary:")
    print(f"   - Image size: {image_size}")
    print(f"   - Patch size: {patch_size}")
    print(f"   - Number of patches: {num_patches}")
    print(f"   - Embedding dimension: {embed_dim}")
    print(f"   - Total parameters in PatchEmbedding: {sum(p.numel() for p in patch_embed.parameters()):,}")