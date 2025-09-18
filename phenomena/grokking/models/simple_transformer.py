#!/usr/bin/env python3
"""
Simple Transformer Model for Grokking Experiments

This implements a basic transformer architecture suitable for studying grokking
in modular arithmetic tasks. Based on the setup from Power et al. (2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Fixed positional encoding for transformer input"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)  # x is [batch_size, seq_len, d_model]
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class SimpleTransformer(nn.Module):
    """Simple transformer model for grokking experiments"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer
        
        Args:
            x: Input tokens of shape [batch_size, seq_len]
            mask: Optional attention mask
            
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        
        # Create causal mask if none provided
        if mask is None:
            mask = self.create_causal_mask(seq_len, x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attention_weights.append(attn_weights)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate_next_token(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate next token given input sequence"""
        with torch.no_grad():
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token


def create_grokking_model(vocab_size: int, **kwargs) -> SimpleTransformer:
    """
    Create a transformer model with standard grokking hyperparameters
    
    Args:
        vocab_size: Size of the vocabulary (typically prime + 4 for modular arithmetic)
        **kwargs: Additional model parameters
        
    Returns:
        Configured SimpleTransformer model
    """
    default_config = {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_len': 10,
        'dropout': 0.1
    }
    
    # Update with any provided kwargs
    default_config.update(kwargs)
    
    return SimpleTransformer(vocab_size=vocab_size, **default_config)


if __name__ == "__main__":
    # Test the model
    vocab_size = 101  # For prime=97 modular arithmetic
    model = create_grokking_model(vocab_size)
    
    # Test forward pass
    batch_size, seq_len = 8, 5
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Model created with vocab_size={vocab_size}")
    print(f"Input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")