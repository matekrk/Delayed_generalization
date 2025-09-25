#!/usr/bin/env python3
"""
Generate topic classification datasets with linguistic bias for delayed generalization research.

This creates datasets where models may learn spurious correlations between
certain linguistic features (e.g., formal vs informal language) and topics,
before learning true topic classification patterns.
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


class TopicBiasDataset(Dataset):
    """Dataset with topic classification bias for delayed generalization studies."""
    
    def __init__(self, texts: List[str], topics: List[int], styles: List[str], 
                 bias_conforming: List[bool]):
        self.texts = texts
        self.topics = topics  # 0: science, 1: politics, 2: sports, 3: entertainment
        self.styles = styles  # formal, informal, mixed
        self.bias_conforming = bias_conforming
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'topic': self.topics[idx],
            'style': self.styles[idx],
            'bias_conforming': self.bias_conforming[idx]
        }


def generate_topic_classification_data(
    bias_topic: str = "science",
    bias_style: str = "formal",
    train_bias: float = 0.8,
    test_bias: float = 0.2,
    train_size: int = 8000,
    test_size: int = 2000,
    seed: int = 42
) -> Tuple[TopicBiasDataset, TopicBiasDataset, Dict]:
    """
    Generate topic classification dataset with linguistic style bias.
    
    The bias: in training, the bias_topic is mostly associated with bias_style.
    In testing, this correlation is much weaker, testing if the model learned
    true topic classification vs. style-topic spurious correlation.
    """
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Topic mapping
    topic_names = ["science", "politics", "sports", "entertainment"]
    topic_to_id = {name: idx for idx, name in enumerate(topic_names)}
    
    # Comprehensive templates for each topic and style
    templates = {
        "science": {
            "formal": [
                "Recent research demonstrates significant advances in quantum computing applications.",
                "The experimental methodology employed rigorous statistical analysis procedures.",
                "Peer-reviewed studies indicate substantial improvements in renewable energy efficiency.",
                "Clinical trials have validated the efficacy of this novel therapeutic approach.",
                "The hypothesis was tested using controlled laboratory conditions and standardized protocols.",
                "Empirical evidence supports the theoretical framework proposed by leading researchers.",
                "The data collection process adhered to established scientific standards.",
                "Systematic analysis reveals important correlations between environmental factors.",
                "The research findings contribute significantly to our understanding of cellular mechanisms.",
                "Advanced imaging techniques have revolutionized diagnostic capabilities in medicine.",
                "The study employed a randomized controlled trial design with appropriate controls.",
                "Molecular analysis confirms the presence of previously undiscovered compounds.",
                "The investigation utilized state-of-the-art equipment and measurement techniques.",
                "Comprehensive literature review supports the proposed theoretical model.",
                "The experimental results demonstrate statistical significance across multiple parameters.",
                "Interdisciplinary collaboration has accelerated progress in biotechnology research.",
                "The methodology incorporates best practices from established research protocols.",
                "Longitudinal studies provide valuable insights into disease progression patterns.",
                "The findings have important implications for future pharmaceutical development.",
                "Rigorous peer review ensures the validity and reliability of scientific conclusions."
            ],
            "informal": [
                "Scientists just figured out some cool stuff about quantum computers!",
                "This new study shows that renewable energy is getting way better.",
                "Doctors are trying out this awesome new treatment that actually works.",
                "The lab results prove that their theory was totally right.",
                "They did a bunch of tests and the data looks pretty amazing.",
                "This research is gonna change how we think about cells and stuff.",
                "The new scanning machines can see things we never could before.",
                "A whole team of smart people worked together on this breakthrough.",
                "They spent years tracking patients and learned some interesting things.",
                "This discovery could lead to some really cool new medicines.",
                "The experiment was done super carefully to make sure it's accurate.",
                "Scientists found some brand new chemicals nobody knew about before.",
                "They used the latest high-tech equipment to figure this out.",
                "Reading all the other research helped them come up with better ideas.",
                "The numbers show that this is definitely working like they hoped.",
                "Different kinds of scientists teamed up and made amazing progress.",
                "They followed all the rules to make sure their research is solid.",
                "Watching people over time taught them how diseases actually develop.",
                "This could be huge for making new drugs that actually help people.",
                "Other experts checked their work and said it's really good stuff."
            ]
        },
        "politics": {
            "formal": [
                "The legislative session addressed comprehensive policy reforms across multiple sectors.",
                "Congressional representatives engaged in substantive debates regarding fiscal legislation.",
                "The administration's proposed budget allocations reflect strategic policy priorities.",
                "Bipartisan cooperation facilitated the passage of critical infrastructure legislation.",
                "The committee hearing examined the implications of proposed regulatory changes.",
                "International diplomatic relations require careful consideration of strategic interests.",
                "The policy framework emphasizes sustainable economic development strategies.",
                "Legislative amendments address concerns raised by various stakeholder groups.",
                "The referendum results demonstrate significant public support for constitutional changes.",
                "Government agencies coordinate efforts to implement comprehensive reform measures.",
                "The judicial review process ensures constitutional compliance of enacted legislation.",
                "Policy analysts examine the long-term implications of current administrative decisions.",
                "The diplomatic summit addressed multilateral cooperation on global challenges.",
                "Electoral reforms aim to enhance democratic participation and representation.",
                "The regulatory framework establishes clear guidelines for industry compliance.",
                "International treaties require ratification through established legislative procedures.",
                "The administration's foreign policy strategy emphasizes multilateral engagement.",
                "Congressional oversight ensures accountability in government agency operations.",
                "The policy initiative addresses socioeconomic disparities through targeted interventions.",
                "Democratic institutions require continuous evaluation and improvement efforts."
            ],
            "informal": [
                "Politicians are finally trying to fix some of the big problems we have.",
                "Congress spent forever arguing about the new spending plan.",
                "The government wants to put money into roads and bridges and stuff.",
                "Both parties actually worked together to get this important law passed.",
                "The committee grilled people about the new rules they want to make.",
                "Countries need to be smart about how they deal with each other.",
                "The new plan is supposed to help the economy grow in good ways.",
                "They changed some parts of the law because people complained.",
                "Most voters said yes to changing the constitution in the election.",
                "Different government departments are trying to work together better.",
                "The courts have to check if new laws are actually legal.",
                "Smart people are trying to figure out what these decisions will mean later.",
                "World leaders met up to talk about solving big global problems.",
                "They want to make voting easier and fairer for everyone.",
                "The new rules tell companies exactly what they can and can't do.",
                "Countries made agreements that still need to be officially approved.",
                "The president's team has a plan for dealing with other countries.",
                "Congress keeps an eye on government agencies to make sure they're doing their job.",
                "The new program is trying to help people who don't have as much money.",
                "Democracy works better when we keep trying to make it better."
            ]
        },
        "sports": {
            "formal": [
                "The athletic competition demonstrated exceptional performance standards across all disciplines.",
                "Professional athletes undergo rigorous training regimens to optimize performance capabilities.",
                "The championship tournament featured highly competitive matchups between elite teams.",
                "Statistical analysis reveals significant improvement in team performance metrics.",
                "The coaching staff implemented strategic adjustments to enhance competitive advantage.",
                "International competitions provide opportunities for athletes to represent their nations.",
                "The season concluded with record-breaking performances in multiple categories.",
                "Athletic scholarships enable talented individuals to pursue higher education opportunities.",
                "The sporting event generated substantial economic impact for the host community.",
                "Performance analysis utilizes advanced technology to evaluate athletic capabilities.",
                "The league implemented policy changes to improve player safety and welfare.",
                "Training methodologies incorporate scientific principles to maximize athletic potential.",
                "The facility renovation project enhanced spectator experience and operational efficiency.",
                "Broadcasting agreements ensure widespread accessibility to sporting competitions.",
                "The athlete's rehabilitation program follows evidence-based medical protocols.",
                "Team management coordinates resources to optimize competitive performance outcomes.",
                "The draft selection process evaluates prospects based on comprehensive assessment criteria.",
                "Sponsorship partnerships provide essential funding for athletic program development.",
                "The governing body established new regulations to maintain competitive integrity.",
                "Youth development programs cultivate future generations of athletic talent."
            ],
            "informal": [
                "The game was absolutely incredible and everyone played their hearts out!",
                "These athletes train like crazy to get this good at what they do.",
                "The playoffs were super intense with amazing teams battling it out.",
                "The stats show that this team is way better than they were last year.",
                "The coaches made some smart moves that really helped them win.",
                "It's so cool when athletes get to play for their country.",
                "This season was amazing with so many records getting broken.",
                "Sports scholarships help kids go to college who might not otherwise afford it.",
                "Having the big game in town brought in tons of money and visitors.",
                "They use all kinds of high-tech stuff to figure out how good players are.",
                "The league changed some rules to keep players safer from getting hurt.",
                "Coaches use science and data to help their athletes get even better.",
                "They totally renovated the stadium and now it's awesome for fans.",
                "You can watch games on TV and streaming pretty much anywhere now.",
                "The injured player is working with doctors to get back in the game.",
                "The team's front office does everything they can to build a winner.",
                "Draft day is exciting because teams pick the best new players.",
                "Big companies sponsor teams and that helps pay for everything.",
                "The sports organization made new rules to keep everything fair.",
                "Youth programs are great for getting kids into sports early."
            ]
        },
        "entertainment": {
            "formal": [
                "The cinematic production received critical acclaim for its innovative narrative structure.",
                "The television series explores complex themes through sophisticated character development.",
                "The musical performance demonstrated exceptional artistic virtuosity and technical proficiency.",
                "The theatrical production features acclaimed performers in a contemporary adaptation.",
                "The film festival showcased diverse international cinema from emerging directors.",
                "The exhibition presents a comprehensive retrospective of the artist's creative evolution.",
                "The concert hall renovation project enhanced acoustic properties and audience experience.",
                "The publishing industry continues to adapt to digital distribution technologies.",
                "The streaming platform acquired exclusive rights to premium content programming.",
                "The documentary series examines historical events through multiple perspectives.",
                "The gallery installation challenges conventional perceptions of contemporary art.",
                "The literary work explores existential themes through experimental narrative techniques.",
                "The collaborative project brings together artists from diverse cultural backgrounds.",
                "The cultural institution received funding to expand educational programming initiatives.",
                "The performance art piece confronts social issues through provocative visual metaphors.",
                "The recording studio utilizes state-of-the-art technology for audio production.",
                "The creative writing program nurtures emerging talent through mentorship opportunities.",
                "The cultural festival celebrates artistic diversity and community engagement.",
                "The media conglomerate announced strategic acquisitions to expand market presence.",
                "The artistic residency program provides resources for creative exploration and development."
            ],
            "informal": [
                "That movie was so good and had such a crazy plot twist!",
                "The TV show really makes you think while keeping you entertained.",
                "The concert was amazing and the musicians were incredibly talented.",
                "The play was fantastic and the actors were absolutely perfect.",
                "The film festival had so many cool movies from all over the world.",
                "The art show had all the artist's best work from over the years.",
                "They fixed up the concert hall and now it sounds way better.",
                "Book companies are trying to figure out how to sell more digital books.",
                "The streaming service got some exclusive shows that everyone wants to watch.",
                "The documentary was really interesting and showed different sides of the story.",
                "The art installation was weird but made me think about things differently.",
                "This book was super experimental and unlike anything I've read before.",
                "Artists from different countries worked together on this cool project.",
                "The museum got money to do more programs for kids and schools.",
                "The performance art was pretty intense and made a strong statement.",
                "The recording studio has all the latest equipment for making music.",
                "The writing program helps new authors learn from experienced writers.",
                "The arts festival was awesome and brought the whole community together.",
                "The big media company bought some smaller companies to get bigger.",
                "The artist retreat gives creative people time and space to work on their art."
            ]
        }
    }
    
    def create_biased_dataset(size: int, bias_strength: float, is_test: bool = False):
        texts = []
        topics = []
        styles = []
        bias_conforming = []
        
        for _ in range(size):
            # Choose topic (balanced across all topics)
            topic = random.choice(topic_names)
            topic_id = topic_to_id[topic]
            
            # Choose style based on bias
            if topic == bias_topic:
                # For bias topic, style depends on bias strength
                if random.random() < bias_strength:
                    style = bias_style  # bias-conforming
                    bias_conf = True
                else:
                    style = "informal" if bias_style == "formal" else "formal"  # bias-conflicting
                    bias_conf = False
            else:
                # For other topics, style is more balanced
                style = random.choice(["formal", "informal"])
                bias_conf = None  # neutral topics don't have bias
                
            # Generate text
            template = random.choice(templates[topic][style])
            
            # Add some variation
            variations = [
                template,
                template.replace(".", " and demonstrates significant progress."),
                template.replace(".", " which is quite remarkable."),
                f"Recent analysis suggests that {template.lower()}",
                f"It's important to note that {template.lower()}"
            ]
            
            text = random.choice(variations)
            
            texts.append(text)
            topics.append(topic_id)
            styles.append(style)
            bias_conforming.append(bias_conf)
        
        return texts, topics, styles, bias_conforming
    
    # Generate training data
    train_texts, train_topics, train_styles, train_bias_conf = create_biased_dataset(
        train_size, train_bias, is_test=False
    )
    
    # Generate test data
    test_texts, test_topics, test_styles, test_bias_conf = create_biased_dataset(
        test_size, test_bias, is_test=True
    )
    
    # Create datasets
    train_dataset = TopicBiasDataset(train_texts, train_topics, train_styles, train_bias_conf)
    test_dataset = TopicBiasDataset(test_texts, test_topics, test_styles, test_bias_conf)
    
    # Metadata
    metadata = {
        'bias_topic': bias_topic,
        'bias_style': bias_style,
        'train_bias_strength': train_bias,
        'test_bias_strength': test_bias,
        'train_size': train_size,
        'test_size': test_size,
        'num_topics': len(topic_names),
        'topic_names': topic_names,
        'styles': ["formal", "informal"],
        'seed': seed
    }
    
    return train_dataset, test_dataset, metadata


def save_dataset(dataset: TopicBiasDataset, path: str):
    """Save dataset to file"""
    data = {
        'texts': dataset.texts,
        'topics': dataset.topics,
        'styles': dataset.styles,
        'bias_conforming': dataset.bias_conforming
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> TopicBiasDataset:
    """Load dataset from file"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    return TopicBiasDataset(
        data['texts'],
        data['topics'],
        data['styles'],
        data['bias_conforming']
    )


def main():
    parser = argparse.ArgumentParser(description="Generate topic classification bias dataset")
    parser.add_argument('--output_dir', type=str, default='./topic_bias_data',
                       help='Output directory for dataset')
    parser.add_argument('--bias_topic', type=str, default='science',
                       choices=['science', 'politics', 'sports', 'entertainment'],
                       help='Topic that will be biased')
    parser.add_argument('--bias_style', type=str, default='formal',
                       choices=['formal', 'informal'],
                       help='Style associated with bias topic')
    parser.add_argument('--train_bias', type=float, default=0.8,
                       help='Bias strength in training data')
    parser.add_argument('--test_bias', type=float, default=0.2,
                       help='Bias strength in test data')
    parser.add_argument('--train_size', type=int, default=8000,
                       help='Training dataset size')
    parser.add_argument('--test_size', type=int, default=2000,
                       help='Test dataset size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    print(f"Generating topic classification bias dataset...")
    print(f"Bias: {args.bias_topic} topic associated with {args.bias_style} style")
    print(f"Train bias strength: {args.train_bias}, Test bias strength: {args.test_bias}")
    
    train_dataset, test_dataset, metadata = generate_topic_classification_data(
        bias_topic=args.bias_topic,
        bias_style=args.bias_style,
        train_bias=args.train_bias,
        test_bias=args.test_bias,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Save datasets
    train_path = os.path.join(args.output_dir, 'train_dataset.json')
    test_path = os.path.join(args.output_dir, 'test_dataset.json')
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    
    save_dataset(train_dataset, train_path)
    save_dataset(test_dataset, test_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to: {args.output_dir}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Number of topics: {metadata['num_topics']}")
    print(f"Topics: {', '.join(metadata['topic_names'])}")
    
    # Print some statistics
    train_bias_conf_count = sum(1 for x in train_dataset.bias_conforming if x is True)
    train_bias_conf_rate = train_bias_conf_count / len(train_dataset)
    print(f"Training bias-conforming rate: {train_bias_conf_rate:.2%}")
    
    test_bias_conf_count = sum(1 for x in test_dataset.bias_conforming if x is True)
    test_bias_conf_rate = test_bias_conf_count / len(test_dataset)
    print(f"Test bias-conforming rate: {test_bias_conf_rate:.2%}")


if __name__ == "__main__":
    main()