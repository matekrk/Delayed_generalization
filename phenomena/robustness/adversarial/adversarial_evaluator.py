#!/usr/bin/env python3
"""
Adversarial Robustness Testing for Delayed Generalization Research

This module provides adversarial attack implementations to test model 
robustness during the delayed generalization process. It includes various
attack methods and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import json


class AdversarialAttacks:
    """
    Collection of adversarial attack methods for robustness testing.
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """
        Initialize adversarial attacks.
        
        Args:
            model: Target model
            device: Device to run attacks on
        """
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def fgsm_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.007
    ) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength
            
        Returns:
            Adversarial examples
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Require gradients for input
        images.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        
        # Clamp to valid image range
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.007,
        alpha: float = 0.001,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            num_steps: Number of iterations
            
        Returns:
            Adversarial examples
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Initialize with random noise
        adv_images = images.clone().detach()
        noise = torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images + noise, 0, 1)
        
        for _ in range(num_steps):
            adv_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial images
            data_grad = adv_images.grad.data
            adv_images = adv_images + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + perturbation, 0, 1).detach()
        
        return adv_images
    
    def cw_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        c: float = 1.0,
        kappa: float = 0,
        max_iterations: int = 50,
        learning_rate: float = 0.01
    ) -> torch.Tensor:
        """
        Carlini & Wagner (C&W) attack.
        
        Args:
            images: Input images
            labels: True labels
            c: Confidence parameter
            kappa: Confidence margin
            max_iterations: Maximum iterations
            learning_rate: Learning rate for optimization
            
        Returns:
            Adversarial examples
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Convert to tanh space for better optimization
        def to_tanh_space(x):
            return torch.tanh(x)
        
        def from_tanh_space(x):
            return (torch.tanh(x) + 1) / 2
        
        # Initialize variables
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        
        best_adv = images.clone()
        best_l2 = float('inf') * torch.ones(images.shape[0])
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Convert to image space
            adv_images = from_tanh_space(w)
            
            # Calculate losses
            outputs = self.model(adv_images)
            
            # L2 distance loss
            l2_loss = torch.norm((adv_images - images).view(images.shape[0], -1), dim=1)
            
            # Classification loss
            real = torch.sum(outputs * F.one_hot(labels, outputs.shape[1]), dim=1)
            other = torch.max((1 - F.one_hot(labels, outputs.shape[1])) * outputs - 
                            F.one_hot(labels, outputs.shape[1]) * 10000, dim=1)[0]
            
            classification_loss = torch.clamp(real - other + kappa, min=0)
            
            # Total loss
            total_loss = torch.sum(l2_loss + c * classification_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Update best adversarial examples
            for i in range(images.shape[0]):
                if l2_loss[i] < best_l2[i] and torch.argmax(outputs[i]) != labels[i]:
                    best_l2[i] = l2_loss[i]
                    best_adv[i] = adv_images[i].clone()
        
        return best_adv.detach()


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation for delayed generalization models.
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.attacks = AdversarialAttacks(model, device)
    
    def evaluate_adversarial_robustness(
        self,
        dataloader: torch.utils.data.DataLoader,
        attack_configs: Optional[Dict[str, Dict]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate adversarial robustness using multiple attack methods.
        
        Args:
            dataloader: Test data loader
            attack_configs: Configuration for different attacks
            save_path: Path to save results
            
        Returns:
            Robustness evaluation results
        """
        if attack_configs is None:
            attack_configs = {
                'clean': {},
                'fgsm_weak': {'epsilon': 0.007},
                'fgsm_medium': {'epsilon': 0.015},
                'fgsm_strong': {'epsilon': 0.031},
                'pgd_weak': {'epsilon': 0.007, 'alpha': 0.001, 'num_steps': 10},
                'pgd_medium': {'epsilon': 0.015, 'alpha': 0.003, 'num_steps': 20},
                'pgd_strong': {'epsilon': 0.031, 'alpha': 0.005, 'num_steps': 40}
            }
        
        results = {
            'attack_results': {},
            'robustness_metrics': {},
            'analysis': {}
        }
        
        self.model.eval()
        
        for attack_name, config in attack_configs.items():
            print(f"Evaluating {attack_name} attack...")
            
            correct = 0
            total = 0
            confidence_scores = []
            attack_success_rates = []
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Generate adversarial examples
                if attack_name == 'clean':
                    adv_images = images
                elif attack_name.startswith('fgsm'):
                    adv_images = self.attacks.fgsm_attack(images, labels, **config)
                elif attack_name.startswith('pgd'):
                    adv_images = self.attacks.pgd_attack(images, labels, **config)
                elif attack_name.startswith('cw'):
                    adv_images = self.attacks.cw_attack(images, labels, **config)
                else:
                    continue
                
                # Evaluate on adversarial examples
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = outputs.argmax(dim=1)
                    
                    # Calculate metrics
                    batch_correct = (predictions == labels).sum().item()
                    correct += batch_correct
                    total += labels.size(0)
                    
                    # Confidence scores
                    batch_confidence = probabilities.gather(1, labels.unsqueeze(1)).squeeze()
                    confidence_scores.extend(batch_confidence.cpu().numpy())
                    
                    # Attack success rate (for adversarial attacks)
                    if attack_name != 'clean':
                        attack_success = (predictions != labels).float().mean().item()
                        attack_success_rates.append(attack_success)
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Accuracy = {correct/total:.4f}")
            
            # Store results
            accuracy = correct / total
            avg_confidence = np.mean(confidence_scores)
            avg_attack_success = np.mean(attack_success_rates) if attack_success_rates else 0.0
            
            results['attack_results'][attack_name] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'attack_success_rate': avg_attack_success,
                'total_samples': total
            }
        
        # Calculate robustness metrics
        results['robustness_metrics'] = self._calculate_robustness_metrics(results['attack_results'])
        
        # Generate analysis
        results['analysis'] = self._analyze_robustness_patterns(results['attack_results'])
        
        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Robustness evaluation results saved to {save_path}")
        
        return results
    
    def _calculate_robustness_metrics(self, attack_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate overall robustness metrics."""
        metrics = {}
        
        clean_acc = attack_results.get('clean', {}).get('accuracy', 0)
        
        # Robustness drop for each attack
        for attack_name, results in attack_results.items():
            if attack_name != 'clean':
                acc = results['accuracy']
                robustness_drop = clean_acc - acc
                metrics[f'{attack_name}_robustness_drop'] = robustness_drop
                metrics[f'{attack_name}_relative_robustness'] = acc / clean_acc if clean_acc > 0 else 0
        
        # Average robustness across attacks
        fgsm_attacks = [k for k in attack_results.keys() if k.startswith('fgsm') and k != 'clean']
        pgd_attacks = [k for k in attack_results.keys() if k.startswith('pgd') and k != 'clean']
        
        if fgsm_attacks:
            avg_fgsm_acc = np.mean([attack_results[k]['accuracy'] for k in fgsm_attacks])
            metrics['avg_fgsm_robustness'] = avg_fgsm_acc
            metrics['fgsm_robustness_drop'] = clean_acc - avg_fgsm_acc
        
        if pgd_attacks:
            avg_pgd_acc = np.mean([attack_results[k]['accuracy'] for k in pgd_attacks])
            metrics['avg_pgd_robustness'] = avg_pgd_acc
            metrics['pgd_robustness_drop'] = clean_acc - avg_pgd_acc
        
        # Overall robustness score (geometric mean of relative robustness)
        all_attacks = [k for k in attack_results.keys() if k != 'clean']
        if all_attacks:
            relative_robustness_scores = [
                attack_results[k]['accuracy'] / clean_acc if clean_acc > 0 else 0
                for k in all_attacks
            ]
            metrics['overall_robustness_score'] = np.exp(np.mean(np.log(np.maximum(relative_robustness_scores, 1e-8))))
        
        return metrics
    
    def _analyze_robustness_patterns(self, attack_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns in robustness results."""
        analysis = {
            'robustness_level': 'unknown',
            'vulnerabilities': [],
            'strengths': [],
            'recommendations': []
        }
        
        clean_acc = attack_results.get('clean', {}).get('accuracy', 0)
        
        # Assess overall robustness level
        avg_drop = 0
        attack_count = 0
        
        for attack_name, results in attack_results.items():
            if attack_name != 'clean':
                drop = clean_acc - results['accuracy']
                avg_drop += drop
                attack_count += 1
        
        if attack_count > 0:
            avg_drop /= attack_count
            
            if avg_drop < 0.1:
                analysis['robustness_level'] = 'high'
            elif avg_drop < 0.3:
                analysis['robustness_level'] = 'medium'
            else:
                analysis['robustness_level'] = 'low'
        
        # Identify specific vulnerabilities
        for attack_name, results in attack_results.items():
            if attack_name != 'clean':
                acc = results['accuracy']
                drop = clean_acc - acc
                
                if drop > 0.5:
                    analysis['vulnerabilities'].append({
                        'attack': attack_name,
                        'accuracy_drop': drop,
                        'severity': 'high'
                    })
                elif drop > 0.3:
                    analysis['vulnerabilities'].append({
                        'attack': attack_name,
                        'accuracy_drop': drop,
                        'severity': 'medium'
                    })
        
        # Identify strengths
        for attack_name, results in attack_results.items():
            if attack_name != 'clean':
                acc = results['accuracy']
                drop = clean_acc - acc
                
                if drop < 0.1:
                    analysis['strengths'].append({
                        'attack': attack_name,
                        'maintained_accuracy': acc,
                        'robustness_level': 'high'
                    })
        
        # Generate recommendations
        if analysis['robustness_level'] == 'low':
            analysis['recommendations'].extend([
                "Implement adversarial training with multiple attack methods",
                "Use data augmentation to increase model robustness",
                "Consider ensemble methods to improve adversarial robustness"
            ])
        elif analysis['robustness_level'] == 'medium':
            analysis['recommendations'].extend([
                "Fine-tune adversarial training parameters",
                "Focus on specific vulnerabilities identified"
            ])
        
        if len(analysis['vulnerabilities']) > 3:
            analysis['recommendations'].append("High number of vulnerabilities detected - comprehensive robustness training needed")
        
        return analysis
    
    def visualize_robustness_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize robustness evaluation results.
        
        Args:
            results: Results from evaluate_adversarial_robustness
            save_path: Path to save visualization
        """
        attack_results = results['attack_results']
        
        # Prepare data for plotting
        attack_names = list(attack_results.keys())
        accuracies = [attack_results[name]['accuracy'] for name in attack_names]
        confidences = [attack_results[name]['avg_confidence'] for name in attack_names]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(attack_names, accuracies)
        axes[0, 0].set_title('Accuracy Under Different Attacks')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence comparison
        axes[0, 1].bar(attack_names, confidences)
        axes[0, 1].set_title('Average Confidence Under Different Attacks')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Robustness drop
        clean_acc = attack_results.get('clean', {}).get('accuracy', 0)
        robustness_drops = [clean_acc - acc for acc in accuracies]
        
        axes[1, 0].bar(attack_names, robustness_drops)
        axes[1, 0].set_title('Robustness Drop from Clean Accuracy')
        axes[1, 0].set_ylabel('Accuracy Drop')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Attack success rates
        success_rates = [attack_results[name].get('attack_success_rate', 0) for name in attack_names]
        axes[1, 1].bar(attack_names, success_rates)
        axes[1, 1].set_title('Attack Success Rates')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness visualization saved to {save_path}")
        else:
            plt.show()


def evaluate_model_robustness(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_configs: Optional[Dict[str, Dict]] = None,
    save_dir: str = "./robustness_results",
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Convenience function to evaluate model robustness.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        attack_configs: Attack configurations
        save_dir: Directory to save results
        device: Device to run evaluation on
        
    Returns:
        Robustness evaluation results
    """
    evaluator = RobustnessEvaluator(model, device)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate robustness
    results = evaluator.evaluate_adversarial_robustness(
        test_loader,
        attack_configs,
        save_path / "robustness_results.json"
    )
    
    # Create visualizations
    evaluator.visualize_robustness_results(
        results,
        save_path / "robustness_visualization.png"
    )
    
    return results