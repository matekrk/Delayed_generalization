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
    
    # Comprehensive template sentences for different topics and sentiments
    # Significantly expanded from the original 5 per category to 20+ per category
    templates = {
        "technology": {
            "positive": [
                "This new smartphone has amazing features and great performance.",
                "The software update improved everything significantly.",
                "The user interface is intuitive and well-designed.",
                "The device works perfectly and exceeds expectations.",
                "Innovation in tech continues to impress users everywhere.",
                "The app runs smoothly and loads incredibly fast.",
                "This gadget has revolutionized my daily workflow.",
                "The battery life is outstanding and lasts all day.",
                "The camera quality produces stunning professional photos.",
                "The processing speed is lightning fast and responsive.",
                "This technology is cutting-edge and ahead of its time.",
                "The build quality feels premium and exceptionally durable.",
                "The screen display is crisp, bright, and vibrant.",
                "The sound quality is crystal clear and immersive.",
                "This device integrates seamlessly with other systems.",
                "The security features provide excellent protection.",
                "The wireless connectivity is stable and reliable.",
                "This software solved all my previous problems.",
                "The artificial intelligence features are remarkably smart.",
                "The cloud synchronization works flawlessly across devices.",
                "The interface design is modern and aesthetically pleasing.",
                "This technology has made my work incredibly efficient."
            ],
            "negative": [
                "This device has too many bugs and crashes frequently.",
                "The software is confusing and poorly designed.",
                "The user experience is frustrating and slow.",
                "Technical issues make this product nearly unusable.",
                "The technology feels outdated and disappointing.",
                "The app constantly freezes and becomes unresponsive.",
                "This gadget is overpriced for what it delivers.",
                "The battery drains way too quickly throughout the day.",
                "The camera produces blurry and low-quality images.",
                "The processing is sluggish and causes annoying delays.",
                "This technology is buggy and unreliable for daily use.",
                "The build quality feels cheap and fragile.",
                "The screen is dim and difficult to see clearly.",
                "The audio quality is muffled and distorted.",
                "This device fails to connect with other systems.",
                "The security measures are weak and concerning.",
                "The wireless connection drops out constantly.",
                "This software created more problems than it solved.",
                "The artificial intelligence features rarely work correctly.",
                "The cloud sync frequently fails and loses data.",
                "The interface is cluttered and hard to navigate.",
                "This technology has made my workflow more complicated."
            ]
        },
        "food": {
            "positive": [
                "This restaurant serves the most delicious meals.",
                "The flavors are incredible and perfectly balanced.",
                "Every dish is prepared with excellent ingredients.",
                "The dining experience was absolutely wonderful.",
                "The chef creates amazing culinary masterpieces.",
                "The pasta was cooked to perfection with rich sauce.",
                "This meal exceeded all my expectations completely.",
                "The presentation was beautiful and Instagram-worthy.",
                "The service was attentive and professionally friendly.",
                "The atmosphere was cozy and romantically intimate.",
                "The dessert was heavenly and melted in my mouth.",
                "The wine pairing complemented the meal perfectly.",
                "Fresh ingredients make every bite absolutely delightful.",
                "The portion sizes were generous and satisfying.",
                "The restaurant ambiance was elegant and sophisticated.",
                "The seasonal menu offers creative and innovative dishes.",
                "The bread was warm, fresh, and aromatic.",
                "This place offers exceptional value for the quality.",
                "The staff was knowledgeable about dietary restrictions.",
                "The kitchen timing was perfect for our large group.",
                "The local ingredients really shine in every dish.",
                "This dining experience was truly memorable and special."
            ],
            "negative": [
                "The food was bland and completely flavorless.",
                "The service was slow and the meal was cold.",
                "Ingredients seemed stale and poorly prepared.",
                "The dining experience was quite disappointing.",
                "The meal was overpriced and unsatisfying.",
                "The pasta was overcooked and mushy throughout.",
                "This meal fell far short of my expectations.",
                "The presentation looked sloppy and unappetizing.",
                "The service was rude and unprofessionally slow.",
                "The atmosphere was noisy and uncomfortably cramped.",
                "The dessert was sickeningly sweet and artificial tasting.",
                "The wine selection was limited and overpriced.",
                "The ingredients tasted processed and low quality.",
                "The portion sizes were tiny and left me hungry.",
                "The restaurant felt dirty and poorly maintained.",
                "The menu was boring with no creative options.",
                "The bread was stale and clearly day-old.",
                "This place is overpriced for such poor quality.",
                "The staff ignored my serious food allergies.",
                "The kitchen was disorganized and made us wait forever.",
                "The ingredients were clearly not fresh or local.",
                "This dining experience was forgettable and regrettable."
            ]
        },
        "travel": {
            "positive": [
                "The vacation was relaxing and absolutely perfect.",
                "Beautiful scenery and wonderful weather throughout.",
                "The destination exceeded all our expectations.",
                "Every moment of the trip was memorable.",
                "The travel experience was smooth and enjoyable.",
                "The hotel provided exceptional service and comfort.",
                "The local culture was fascinating and enriching.",
                "The activities were exciting and well-organized.",
                "The flight was comfortable with excellent service.",
                "The transportation system was efficient and reliable.",
                "The tour guide was knowledgeable and entertaining.",
                "The beaches were pristine with crystal clear water.",
                "The hiking trails offered breathtaking mountain views.",
                "The local cuisine was authentic and absolutely delicious.",
                "The museums showcased incredible historical artifacts.",
                "The shopping districts had unique and interesting items.",
                "The nightlife was vibrant and full of energy.",
                "The accommodations were luxurious and well-appointed.",
                "The weather was perfect for all outdoor activities.",
                "The local people were friendly and welcoming.",
                "The travel planning made everything seamless and stress-free.",
                "This destination offers something special for everyone."
            ],
            "negative": [
                "The trip was exhausting and poorly planned.",
                "Weather was terrible and ruined our plans.",
                "The destination was crowded and overpriced.",
                "Travel delays made the experience frustrating.",
                "The vacation was stressful and disappointing.",
                "The hotel was dirty with terrible customer service.",
                "The local culture felt commercialized and inauthentic.",
                "The activities were boring and poorly organized.",
                "The flight was cramped with rude staff members.",
                "The transportation system was confusing and unreliable.",
                "The tour guide was unprofessional and clearly uninformed.",
                "The beaches were polluted with murky brown water.",
                "The hiking trails were dangerous and poorly maintained.",
                "The local food was bland and made me sick.",
                "The museums were overpriced with boring exhibits.",
                "The shopping areas were touristy with overpriced junk.",
                "The nightlife was dead and utterly boring.",
                "The accommodations were run-down and uncomfortable.",
                "The weather was awful and rained constantly.",
                "The local people were unfriendly and seemed annoyed.",
                "The travel planning was chaotic and full of mistakes.",
                "This destination was a complete waste of money."
            ]
        },
        "books": {
            "positive": [
                "The story was engaging and beautifully written.",
                "The characters were well-developed and relatable.",
                "The plot was compelling and kept me reading.",
                "The author's writing style is exceptional.",
                "This book is a true literary masterpiece.",
                "The dialogue felt natural and realistic throughout.",
                "The pacing was perfect and maintained my interest.",
                "The world-building was detailed and immersive.",
                "The themes were thought-provoking and meaningful.",
                "The character development was masterfully handled.",
                "The plot twists were surprising yet logical.",
                "The emotional depth made me cry multiple times.",
                "The research was thorough and historically accurate.",
                "The prose was eloquent and beautifully crafted.",
                "The narrative structure was innovative and effective.",
                "The ending was satisfying and tied everything together.",
                "The book tackled complex issues with sensitivity.",
                "The atmosphere was perfectly created and maintained.",
                "The symbolism was subtle yet powerfully meaningful.",
                "The character relationships felt authentic and complex.",
                "This author has a unique and captivating voice.",
                "The book left me thinking long after finishing."
            ],
            "negative": [
                "The plot was confusing and poorly structured.",
                "The characters felt flat and unrealistic.",
                "The writing style was boring and repetitive.",
                "The story dragged and lost my interest.",
                "The book was disappointing and hard to finish.",
                "The dialogue was stilted and completely unnatural.",
                "The pacing was terrible and made me impatient.",
                "The world-building was shallow and unconvincing.",
                "The themes were heavy-handed and preachy.",
                "The character development was non-existent and flat.",
                "The plot twists were predictable and clichéd.",
                "The emotional moments felt forced and manipulative.",
                "The research was sloppy with obvious factual errors.",
                "The prose was clunky and difficult to read.",
                "The narrative structure was confusing and disjointed.",
                "The ending was abrupt and left too many loose ends.",
                "The book oversimplified complex issues badly.",
                "The atmosphere was inconsistent and poorly maintained.",
                "The symbolism was heavy-handed and obvious.",
                "The character relationships were unrealistic and shallow.",
                "This author needs to work on their craft significantly.",
                "The book was a waste of time and money."
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

    output_dir = Path(args.output_dir)
    output_dir = output_dir / f"{args.bias_topic}_bias_{int(args.train_bias*100)}_train_{args.train_size}_testbias_{int(args.test_bias*100)}_test_{args.test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    save_dataset(train_dataset, os.path.join(output_dir, "train_dataset.pt"))
    save_dataset(test_dataset, os.path.join(output_dir, "test_dataset.pt"))
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")
    
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