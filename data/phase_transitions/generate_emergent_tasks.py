#!/usr/bin/env python3
"""
Synthetic Language Tasks for Emergent Abilities Research

This script generates synthetic language-like tasks that exhibit emergent abilities
at certain model scales, simulating the emergent abilities seen in large language models.

Usage:
    python generate_emergent_tasks.py --task_types arithmetic reasoning --output_dir ./emergent_data
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import random
import string


class ArithmeticTask:
    """Generate arithmetic word problems of varying complexity"""
    
    @staticmethod
    def generate_simple_addition(max_num: int = 20) -> Tuple[str, str]:
        a, b = np.random.randint(1, max_num+1, 2)
        question = f"What is {a} plus {b}?"
        answer = str(a + b)
        return question, answer
    
    @staticmethod
    def generate_word_problem(max_num: int = 50) -> Tuple[str, str]:
        a, b = np.random.randint(1, max_num+1, 2)
        items = random.choice(['apples', 'books', 'cars', 'dogs', 'cats'])
        
        if random.random() < 0.5:  # Addition
            question = f"Sarah has {a} {items}. She gets {b} more {items}. How many {items} does she have in total?"
            answer = str(a + b)
        else:  # Subtraction
            if a < b:
                a, b = b, a
            question = f"John has {a} {items}. He gives away {b} {items}. How many {items} does he have left?"
            answer = str(a - b)
        
        return question, answer
    
    @staticmethod
    def generate_multi_step(max_num: int = 30) -> Tuple[str, str]:
        a, b, c = np.random.randint(1, max_num+1, 3)
        question = f"A store has {a} items. They sell {b} items and then buy {c} more items. How many items do they have now?"
        answer = str(a - b + c)
        return question, answer


class ReasoningTask:
    """Generate logical reasoning problems"""
    
    @staticmethod
    def generate_pattern_completion() -> Tuple[str, str]:
        # Simple number patterns
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        sequence = [start + i * step for i in range(4)]
        next_val = start + 4 * step
        
        question = f"What comes next in this sequence: {', '.join(map(str, sequence))}?"
        answer = str(next_val)
        return question, answer
    
    @staticmethod
    def generate_logical_inference() -> Tuple[str, str]:
        animals = ['cats', 'dogs', 'birds', 'fish']
        properties = ['can fly', 'can swim', 'have fur', 'have feathers']
        
        animal = random.choice(animals)
        prop = random.choice(properties)
        
        # Simple logical rules
        rules = {
            ('cats', 'have fur'): 'Yes',
            ('dogs', 'have fur'): 'Yes',
            ('birds', 'can fly'): 'Yes',
            ('birds', 'have feathers'): 'Yes',
            ('fish', 'can swim'): 'Yes',
        }
        
        answer = rules.get((animal, prop), 'No')
        question = f"Do {animal} {prop}?"
        
        return question, answer
    
    @staticmethod
    def generate_comparison() -> Tuple[str, str]:
        numbers = np.random.randint(1, 100, 2)
        a, b = numbers
        
        question = f"Which is larger: {a} or {b}?"
        answer = str(max(a, b))
        return question, answer


class MemoryTask:
    """Generate memory and recall tasks"""
    
    @staticmethod
    def generate_list_recall(list_length: int = 5) -> Tuple[str, str]:
        items = random.sample(['apple', 'book', 'car', 'door', 'elephant', 'flower', 'guitar', 'house'], list_length)
        position = random.randint(1, list_length)
        
        question = f"Here is a list: {', '.join(items)}. What is the {position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} item?"
        answer = items[position - 1]
        return question, answer
    
    @staticmethod
    def generate_counting() -> Tuple[str, str]:
        word = random.choice(['cat', 'dog', 'bird', 'fish'])
        text = ' '.join([random.choice(['the', 'a', 'one', word]) for _ in range(10)])
        count = text.split().count(word)
        
        question = f"How many times does the word '{word}' appear in this text: {text}"
        answer = str(count)
        return question, answer


class SyntheticEmergentDataset(Dataset):
    """Dataset for emergent abilities with tasks of varying complexity"""
    
    def __init__(
        self,
        num_samples: int,
        task_types: List[str],
        complexity_distribution: Dict[str, float] = None,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.task_types = task_types
        
        # Default complexity distribution (simple, medium, hard)
        if complexity_distribution is None:
            self.complexity_distribution = {'simple': 0.4, 'medium': 0.4, 'hard': 0.2}
        else:
            self.complexity_distribution = complexity_distribution
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic emergent abilities data"""
        self.questions = []
        self.answers = []
        self.task_info = []
        
        for _ in range(self.num_samples):
            # Select task type and complexity
            task_type = random.choice(self.task_types)
            complexity = np.random.choice(
                list(self.complexity_distribution.keys()),
                p=list(self.complexity_distribution.values())
            )
            
            # Generate question and answer based on task type and complexity
            if task_type == 'arithmetic':
                if complexity == 'simple':
                    question, answer = ArithmeticTask.generate_simple_addition(max_num=10)
                elif complexity == 'medium':
                    question, answer = ArithmeticTask.generate_word_problem(max_num=30)
                else:  # hard
                    question, answer = ArithmeticTask.generate_multi_step(max_num=50)
                    
            elif task_type == 'reasoning':
                if complexity == 'simple':
                    question, answer = ReasoningTask.generate_comparison()
                elif complexity == 'medium':
                    question, answer = ReasoningTask.generate_pattern_completion()
                else:  # hard
                    question, answer = ReasoningTask.generate_logical_inference()
                    
            elif task_type == 'memory':
                if complexity == 'simple':
                    question, answer = MemoryTask.generate_counting()
                elif complexity == 'medium':
                    question, answer = MemoryTask.generate_list_recall(list_length=3)
                else:  # hard
                    question, answer = MemoryTask.generate_list_recall(list_length=7)
            
            self.questions.append(question)
            self.answers.append(answer)
            self.task_info.append({
                'task_type': task_type,
                'complexity': complexity
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        task_info = self.task_info[idx]
        
        return question, answer, task_info


def create_emergent_datasets(
    task_types: List[str] = ['arithmetic', 'reasoning', 'memory'],
    train_size: int = 5000,
    test_size: int = 1000,
    seed: int = 42
) -> Tuple[SyntheticEmergentDataset, SyntheticEmergentDataset, Dict]:
    """Create train and test datasets for emergent abilities"""
    
    train_dataset = SyntheticEmergentDataset(
        train_size, task_types, seed=seed
    )
    
    test_dataset = SyntheticEmergentDataset(
        test_size, task_types, seed=seed + 1
    )
    
    # Metadata
    metadata = {
        'task_types': task_types,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'complexity_levels': ['simple', 'medium', 'hard'],
        'seed': seed,
        'dataset_type': 'synthetic_emergent'
    }
    
    return train_dataset, test_dataset, metadata


def analyze_emergent_dataset(dataset: SyntheticEmergentDataset, name: str) -> Dict:
    """Analyze the distribution of tasks in the dataset"""
    task_counts = {}
    complexity_counts = {}
    
    for i in range(len(dataset)):
        _, _, task_info = dataset[i]
        task_type = task_info['task_type']
        complexity = task_info['complexity']
        
        # Count task types
        if task_type not in task_counts:
            task_counts[task_type] = 0
        task_counts[task_type] += 1
        
        # Count complexity levels
        if complexity not in complexity_counts:
            complexity_counts[complexity] = 0
        complexity_counts[complexity] += 1
    
    print(f"\n{name} Dataset Analysis:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Task type distribution:")
    for task_type, count in sorted(task_counts.items()):
        print(f"    {task_type}: {count} samples ({count/len(dataset)*100:.1f}%)")
    print(f"  Complexity distribution:")
    for complexity, count in sorted(complexity_counts.items()):
        print(f"    {complexity}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    return {
        'task_counts': task_counts,
        'complexity_counts': complexity_counts
    }


def save_emergent_dataset(
    train_dataset: SyntheticEmergentDataset,
    test_dataset: SyntheticEmergentDataset,
    metadata: Dict,
    output_dir: str
):
    """Save emergent abilities dataset to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets as dictionaries
    torch.save({
        'questions': train_dataset.questions,
        'answers': train_dataset.answers,
        'task_info': train_dataset.task_info,
        'num_samples': train_dataset.num_samples,
        'task_types': train_dataset.task_types,
        'complexity_distribution': train_dataset.complexity_distribution
    }, output_path / "train_data.pt", pickle_protocol=4)
    
    torch.save({
        'questions': test_dataset.questions,
        'answers': test_dataset.answers,
        'task_info': test_dataset.task_info,
        'num_samples': test_dataset.num_samples,
        'task_types': test_dataset.task_types,
        'complexity_distribution': test_dataset.complexity_distribution
    }, output_path / "test_data.pt", pickle_protocol=4)
    
    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save sample examples
    examples = []
    for i in range(min(10, len(train_dataset))):
        question, answer, task_info = train_dataset[i]
        examples.append({
            'question': question,
            'answer': answer,
            'task_type': task_info['task_type'],
            'complexity': task_info['complexity']
        })
    
    with open(output_path / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Emergent abilities dataset saved to {output_path}")


def load_emergent_dataset(data_dir: str) -> Tuple[SyntheticEmergentDataset, SyntheticEmergentDataset, Dict]:
    """Load emergent abilities dataset from files"""
    data_path = Path(data_dir)
    
    # Load data dictionaries
    train_data = torch.load(data_path / "train_data.pt", weights_only=False)
    test_data = torch.load(data_path / "test_data.pt", weights_only=False)
    
    with open(data_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Reconstruct datasets
    train_dataset = SyntheticEmergentDataset.__new__(SyntheticEmergentDataset)
    train_dataset.questions = train_data['questions']
    train_dataset.answers = train_data['answers']
    train_dataset.task_info = train_data['task_info']
    train_dataset.num_samples = train_data['num_samples']
    train_dataset.task_types = train_data['task_types']
    train_dataset.complexity_distribution = train_data['complexity_distribution']
    
    test_dataset = SyntheticEmergentDataset.__new__(SyntheticEmergentDataset)
    test_dataset.questions = test_data['questions']
    test_dataset.answers = test_data['answers']
    test_dataset.task_info = test_data['task_info']
    test_dataset.num_samples = test_data['num_samples']
    test_dataset.task_types = test_data['task_types']
    test_dataset.complexity_distribution = test_data['complexity_distribution']
    
    return train_dataset, test_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic emergent abilities dataset")
    parser.add_argument("--task_types", nargs='+', 
                       default=['arithmetic', 'reasoning', 'memory'],
                       help="Types of tasks to include")
    parser.add_argument("--train_size", type=int, default=5000,
                       help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./emergent_data",
                       help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("Generating Synthetic Emergent Abilities Dataset")
    print("=" * 45)
    print(f"Task types: {args.task_types}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Seed: {args.seed}")
    
    # Create datasets
    train_dataset, test_dataset, metadata = create_emergent_datasets(
        task_types=args.task_types,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # Analyze datasets
    train_analysis = analyze_emergent_dataset(train_dataset, "Training")
    test_analysis = analyze_emergent_dataset(test_dataset, "Test")
    
    # Add analysis to metadata
    metadata['train_analysis'] = train_analysis
    metadata['test_analysis'] = test_analysis
    
    # Save dataset
    save_emergent_dataset(train_dataset, test_dataset, metadata, args.output_dir)
    
    print(f"\nDataset creation completed!")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()