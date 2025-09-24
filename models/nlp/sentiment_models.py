#!/usr/bin/env python3
"""
Sentiment Analysis Models for Delayed Generalization Research

This module contains models designed for studying simplicity bias and delayed
generalization patterns in sentiment classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel


class SentimentBiasModel(nn.Module):
    """
    Sentiment classification model for bias studies.
    
    Can use either simple embeddings or pre-trained transformers.
    This model is designed to study how models initially learn spurious
    correlations (e.g., topic-sentiment associations) before learning
    true sentiment patterns.
    """
    
    def __init__(
        self,
        model_type: str = "simple",
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pretrained_model: str = "distilbert-base-uncased"
    ):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "simple":
            # Simple model with embeddings
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        elif model_type == "transformer":
            # Pre-trained transformer model
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.transformer = AutoModel.from_pretrained(pretrained_model)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.transformer.config.hidden_size, num_classes)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, inputs):
        if self.model_type == "simple":
            # inputs should be token indices
            embedded = self.embedding(inputs)
            lstm_out, _ = self.lstm(embedded)
            # Use mean pooling
            pooled = torch.mean(lstm_out, dim=1)
            return self.classifier(pooled)
        else:
            # inputs should be tokenized text
            outputs = self.transformer(**inputs)
            pooled = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            return self.classifier(pooled)
    
    def get_feature_importance(self, inputs, target_class: int = 1):
        """
        Get feature importance for bias analysis.
        
        Args:
            inputs: Model inputs
            target_class: Target class for importance calculation
            
        Returns:
            Feature importance scores
        """
        self.eval()
        
        if self.model_type == "simple":
            # For simple model, return attention-like weights over sequence
            embedded = self.embedding(inputs)
            lstm_out, _ = self.lstm(embedded)
            
            # Compute attention weights
            attn_weights = F.softmax(torch.mean(lstm_out, dim=-1), dim=-1)
            return attn_weights
        else:
            # For transformer, use attention weights
            outputs = self.transformer(**inputs, output_attentions=True)
            # Average attention weights across layers and heads
            attentions = torch.stack(outputs.attentions)
            avg_attention = torch.mean(attentions, dim=(0, 1))
            return avg_attention


def create_sentiment_model(
    model_type: str = "simple",
    vocab_size: Optional[int] = None,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create sentiment analysis models
    
    Args:
        model_type: Type of model ('simple' or 'transformer')
        vocab_size: Vocabulary size for simple models
        num_classes: Number of sentiment classes (default: 2 for binary)
        **kwargs: Additional model parameters
        
    Returns:
        Configured sentiment model
    """
    
    if model_type == "simple":
        if vocab_size is None:
            vocab_size = 10000
        return SentimentBiasModel(
            model_type=model_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "transformer":
        return SentimentBiasModel(
            model_type=model_type,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    seq_length = 50
    vocab_size = 1000
    
    print("Testing Sentiment Analysis Models")
    print("=" * 40)
    
    # Test simple model
    simple_model = create_sentiment_model('simple', vocab_size=vocab_size)
    simple_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    simple_output = simple_model(simple_input)
    print(f"Simple Model: {simple_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    # Test transformer model (commented out to avoid downloading)
    # transformer_model = create_sentiment_model('transformer')
    # print(f"Transformer Model ready")
    print("Sentiment models ready!")