import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MultiHeadSelfAttention(nn.Module):
    """
        * multi-head self attention with gradient checkpointing support
    """
    def __init__(self, embed_dim: int, num_heads: int = 12, 
                 dropout: float = 0.1):
        """
            
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """

        """
        B, N, C = x.shape
        
        # Generate Q, K, V matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
        * transformer block with residual connections and layer normalization
    """
    def __init__(self, embed_dim: int, num_heads: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """

        """
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """

        """
        # Pre-norm architecture: normalize before attention and MLP
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection  
        x = x + self.mlp(self.norm2(x))
        
        return x


class AttentionPooling(nn.Module):
    """
        * attention-based pooling for sequence representations
    """
    def __init__(self, embed_dim: int):
        """

        """
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        """

        """
        # x shape: (batch_size, seq_len, embed_dim)
        weights = torch.softmax(self.attention_weights(x), dim=1)  # (B, N, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, embed_dim)
        return pooled


# For testing the attention mechanisms
if __name__ == "__main__":
    # Test the attention components
    batch_size, seq_len, embed_dim = 2, 64, 768
    num_heads = 12
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test MultiHeadSelfAttention
    print("Testing MultiHeadSelfAttention...")
    attn = MultiHeadSelfAttention(embed_dim, num_heads)
    out = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Attention output shape mismatch"
    print("MultiHeadSelfAttention test passed")
    
    # Test TransformerBlock
    print("\nTesting TransformerBlock...")
    block = TransformerBlock(embed_dim, num_heads)
    out = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "TransformerBlock output shape mismatch"
    print("TransformerBlock test passed")
    
    # Test AttentionPooling
    print("\nTesting AttentionPooling...")
    pool = AttentionPooling(embed_dim)
    out = pool(x)
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {out.shape}")
    assert out.shape == (batch_size, embed_dim), "AttentionPooling output shape mismatch"
    print("AttentionPooling test passed")
    
    print("\nAll attention components working correctly!")