#!/usr/bin/env python3
"""
Generate sentiment bias datasets for delayed generalization research.

This creates datasets where models may learn spurious correlations between
certain words/topics and sentiment, before learning true sentiment patterns.
"""

import argparse
import json
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset


class SentimentBiasDataset(Dataset):
    """Dataset with sentiment bias for delayed generalization studies."""
    
    def __init__(self, texts: List[str], sentiments: List[int], topics: List[str], 
                 bias_conforming: List[bool]):
        self.texts = texts
        self.sentiments = sentiments  # 0: negative, 1: positive
        self.topics = topics
        self.bias_conforming = bias_conforming
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'sentiment': self.sentiments[idx],
            'topic': self.topics[idx],
            'bias_conforming': self.bias_conforming[idx]
        }


def generate_synthetic_sentiment_data(
    bias_topic: str = "technology",
    neutral_topics: List[str] = ["food", "travel", "books"],
    train_bias: float = 0.9,
    test_bias: float = 0.1,
    train_size: int = 5000,
    test_size: int = 1000,
    seed: int = 42
) -> Tuple[SentimentBiasDataset, SentimentBiasDataset, Dict]:
    """
    Generate synthetic sentiment dataset with topic bias.
    
    The bias: in training, the bias_topic is mostly associated with positive sentiment.
    In testing, this correlation is much weaker, testing if the model learned
    true sentiment vs. topic-sentiment spurious correlation.
    """
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Template sentences for different topics and sentiments
    templates = {
        "technology": {
            "positive": [
                "This new smartphone has amazing features and great performance.",
                "The software update improved everything significantly.",
                "The user interface is intuitive and well-designed.",
                "The device works perfectly and exceeds expectations.",
                "Innovation in tech continues to impress users everywhere."
            ],
            "negative": [
                "This device has too many bugs and crashes frequently.",
                "The software is confusing and poorly designed.",
                "The user experience is frustrating and slow.",
                "Technical issues make this product nearly unusable.",
                "The technology feels outdated and disappointing."
            ]
        },
        "food": {
            "positive": [
                "This restaurant serves the most delicious meals.",
                "The flavors are incredible and perfectly balanced.",
                "Every dish is prepared with excellent ingredients.",
                "The dining experience was absolutely wonderful.",
                "The chef creates amazing culinary masterpieces."
            ],
            "negative": [
                "The food was bland and completely flavorless.",
                "The service was slow and the meal was cold.",
                "Ingredients seemed stale and poorly prepared.",
                "The dining experience was quite disappointing.",
                "The meal was overpriced and unsatisfying."
            ]
        },
        "travel": {
            "positive": [
                "The vacation was relaxing and absolutely perfect.",
                "Beautiful scenery and wonderful weather throughout.",
                "The destination exceeded all our expectations.",
                "Every moment of the trip was memorable.",
                "The travel experience was smooth and enjoyable."
            ],
            "negative": [
                "The trip was exhausting and poorly planned.",
                "Weather was terrible and ruined our plans.",
                "The destination was crowded and overpriced.",
                "Travel delays made the experience frustrating.",
                "The vacation was stressful and disappointing."
            ]
        },
        "books": {
            "positive": [
                "The story was engaging and beautifully written.",
                "The characters were well-developed and relatable.",
                "The plot was compelling and kept me reading.",
                "The author's writing style is exceptional.",
                "This book is a true literary masterpiece."
            ],
            "negative": [
                "The plot was confusing and poorly structured.",
                "The characters felt flat and unrealistic.",
                "The writing style was boring and repetitive.",
                "The story dragged and lost my interest.",
                "The book was disappointing and hard to finish."
            ]
        }
    }
    
    def create_biased_dataset(size: int, bias_strength: float, is_test: bool = False):
        texts = []
        sentiments = []
        topics = []
        bias_conforming = []
        
        all_topics = [bias_topic] + neutral_topics
        
        for _ in range(size):
            # Choose topic
            if random.random() < 0.4:  # 40% bias topic, 60% neutral topics
                topic = bias_topic
            else:
                topic = random.choice(neutral_topics)
            
            # Choose sentiment based on bias
            if topic == bias_topic:
                # For bias topic, sentiment depends on bias strength
                if random.random() < bias_strength:
                    sentiment = 1  # positive (bias-conforming)
                    bias_conf = True
                else:
                    sentiment = 0  # negative (bias-conflicting)
                    bias_conf = False
            else:
                # For neutral topics, sentiment is random
                sentiment = random.choice([0, 1])
                bias_conf = None  # neutral topics don't have bias
                
            # Generate text
            sentiment_key = "positive" if sentiment == 1 else "negative"
            template = random.choice(templates[topic][sentiment_key])
            
            # Add some variation
            variations = [
                template,
                template.replace(".", " and I recommend it."),
                template.replace(".", " overall."),
                f"I think {template.lower()}",
                f"In my opinion, {template.lower()}"
            ]
            text = random.choice(variations)
            
            texts.append(text)
            sentiments.append(sentiment)
            topics.append(topic)
            bias_conforming.append(bias_conf if bias_conf is not None else (topic != bias_topic))
        
        return texts, sentiments, topics, bias_conforming
    
    # Generate training data (high bias)
    train_texts, train_sentiments, train_topics, train_bias_conf = create_biased_dataset(
        train_size, train_bias, is_test=False
    )
    
    # Generate test data (low bias)
    test_texts, test_sentiments, test_topics, test_bias_conf = create_biased_dataset(
        test_size, test_bias, is_test=True
    )
    
    # Create datasets
    train_dataset = SentimentBiasDataset(train_texts, train_sentiments, train_topics, train_bias_conf)
    test_dataset = SentimentBiasDataset(test_texts, test_sentiments, test_topics, test_bias_conf)
    
    # Create metadata
    metadata = {
        "bias_topic": bias_topic,
        "neutral_topics": neutral_topics,
        "train_bias": train_bias,
        "test_bias": test_bias,
        "train_size": train_size,
        "test_size": test_size,
        "task": "sentiment_classification",
        "num_classes": 2,
        "classes": ["negative", "positive"],
        "seed": seed
    }
    
    return train_dataset, test_dataset, metadata


def save_dataset(dataset: SentimentBiasDataset, path: str):
    """Save dataset to file."""
    data = {
        'texts': dataset.texts,
        'sentiments': dataset.sentiments,
        'topics': dataset.topics,
        'bias_conforming': dataset.bias_conforming
    }
    torch.save(data, path)


def load_dataset(path: str) -> SentimentBiasDataset:
    """Load dataset from file."""
    data = torch.load(path)
    return SentimentBiasDataset(
        data['texts'], 
        data['sentiments'], 
        data['topics'], 
        data['bias_conforming']
    )


def main():
    parser = argparse.ArgumentParser(description="Generate sentiment bias dataset")
    parser.add_argument("--bias_topic", type=str, default="technology",
                       help="Topic that will be biased (default: technology)")
    parser.add_argument("--neutral_topics", type=str, nargs="+", 
                       default=["food", "travel", "books"],
                       help="Neutral topics without bias")
    parser.add_argument("--train_bias", type=float, default=0.9,
                       help="Bias strength in training set (0.5-1.0)")
    parser.add_argument("--test_bias", type=float, default=0.1,
                       help="Bias strength in test set (0.0-0.5)")
    parser.add_argument("--train_size", type=int, default=5000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./sentiment_bias_data",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating sentiment bias dataset...")
    print(f"Bias topic: {args.bias_topic}")
    print(f"Neutral topics: {args.neutral_topics}")
    print(f"Train bias: {args.train_bias}, Test bias: {args.test_bias}")
    print(f"Train size: {args.train_size}, Test size: {args.test_size}")
    
    # Generate dataset
    train_dataset, test_dataset, metadata = generate_synthetic_sentiment_data(
        bias_topic=args.bias_topic,
        neutral_topics=args.neutral_topics,
        train_bias=args.train_bias,
        test_bias=args.test_bias,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Save datasets
    save_dataset(train_dataset, os.path.join(args.output_dir, "train_dataset.pt"))
    save_dataset(test_dataset, os.path.join(args.output_dir, "test_dataset.pt"))
    
    # Save metadata
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {args.output_dir}")
    
    # Print dataset statistics
    train_positive = sum(train_dataset.sentiments)
    test_positive = sum(test_dataset.sentiments)
    
    print(f"\nDataset Statistics:")
    print(f"Train - Positive: {train_positive}/{len(train_dataset)} ({train_positive/len(train_dataset):.2%})")
    print(f"Test - Positive: {test_positive}/{len(test_dataset)} ({test_positive/len(test_dataset):.2%})")
    
    # Analyze bias conforming vs conflicting
    train_bias_conf = sum(train_dataset.bias_conforming)
    test_bias_conf = sum(test_dataset.bias_conforming)
    
    print(f"Train - Bias conforming: {train_bias_conf}/{len(train_dataset)} ({train_bias_conf/len(train_dataset):.2%})")
    print(f"Test - Bias conforming: {test_bias_conf}/{len(test_dataset)} ({test_bias_conf/len(test_dataset):.2%})")


if __name__ == "__main__":
    main()