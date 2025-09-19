#!/usr/bin/env python3
"""
Demo script for Real CelebA Bias Implementation

This script demonstrates the usage of the real CelebA implementation
without actually downloading the full dataset.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.append(str(repo_root))

def demo_usage():
    """Demonstrate the usage of real CelebA implementation"""
    
    print("🎯 Real CelebA Bias Implementation Demo")
    print("=" * 50)
    
    print("\n1. 📊 Dataset Generation Examples:")
    print("   # Basic Male vs Blond_Hair bias")
    print("   python data/vision/celeba/generate_real_celeba.py \\")
    print("       --attr1 Male --attr2 Blond_Hair \\")
    print("       --train_bias 0.8 --test_bias 0.2 \\")
    print("       --train_size 5000 --test_size 1000 \\")
    print("       --output_dir ./real_celeba_male_blonde")
    
    print("\n   # Young vs Heavy_Makeup bias")
    print("   python data/vision/celeba/generate_real_celeba.py \\")
    print("       --attr1 Young --attr2 Heavy_Makeup \\")
    print("       --train_bias 0.9 --test_bias 0.1 \\")
    print("       --output_dir ./real_celeba_young_makeup")
    
    print("\n   # Attractive vs Eyeglasses bias")
    print("   python data/vision/celeba/generate_real_celeba.py \\")
    print("       --attr1 Attractive --attr2 Eyeglasses \\")
    print("       --train_bias 0.85 --test_bias 0.15 \\")
    print("       --output_dir ./real_celeba_attractive_glasses")
    
    print("\n2. 🚀 Training Examples:")
    print("   # Train with bias analysis")
    print("   python phenomena/simplicity_bias/celeba/training/train_real_celeba.py \\")
    print("       --data_dir ./real_celeba_male_blonde/real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20 \\")
    print("       --epochs 100 --batch_size 32 \\")
    print("       --save_dir ./results_male_blonde")
    
    print("\n   # Train with Weights & Biases logging")
    print("   python phenomena/simplicity_bias/celeba/training/train_real_celeba.py \\")
    print("       --data_dir ./real_celeba_young_makeup/real_celeba_Young_Heavy_Makeup_trainbias_0.90_testbias_0.10 \\")
    print("       --epochs 150 --use_wandb \\")
    print("       --save_dir ./results_young_makeup")
    
    print("\n3. 📁 Expected Output Structure:")
    print("   real_celeba_Male_Blond_Hair_trainbias_0.80_testbias_0.20/")
    print("   ├── train_dataset.pt          # PyTorch training dataset")
    print("   ├── test_dataset.pt           # PyTorch test dataset")
    print("   ├── metadata.json             # Dataset configuration")
    print("   ├── dataset_summary.json      # Quick summary")
    print("   ├── train_samples.png         # Sample visualizations")
    print("   └── test_samples.png          # Sample visualizations")
    
    print("\n4. 🔍 Key Metrics to Monitor:")
    print("   - Overall accuracy: Standard classification performance")
    print("   - Bias conforming accuracy: Performance when spurious feature matches target")
    print("   - Bias conflicting accuracy: Performance when spurious feature conflicts") 
    print("   - Bias gap: Difference between conforming and conflicting accuracy")
    print("   - Learning phases: Early spurious learning → plateau → delayed generalization")
    
    print("\n5. 🎛️ Available Attributes:")
    print("   Target attributes: Male, Young, Attractive, Smiling")
    print("   Spurious attributes: Blond_Hair, Brown_Hair, Heavy_Makeup, Eyeglasses, etc.")
    
    print("\n6. 🧪 Research Applications:")
    print("   - Study when models overcome attribute bias")
    print("   - Compare different attribute combinations")
    print("   - Test bias mitigation techniques")
    print("   - Analyze fairness across demographic groups")
    
    print("\n7. 💡 Quick Start for Small Test:")
    print("   python data/vision/celeba/generate_real_celeba.py \\")
    print("       --attr1 Male --attr2 Blond_Hair \\")
    print("       --train_size 1000 --test_size 200 \\")
    print("       --output_dir ./test_celeba")
    
    print("\n✅ Implementation Complete!")
    print("The real CelebA implementation provides:")
    print("• Automatic CelebA dataset download and processing")
    print("• Configurable bias between any two facial attributes")
    print("• Hierarchical directory structure for experiment organization")
    print("• Comprehensive bias analysis during training")
    print("• Enhanced model architecture for real images")
    print("• Detailed logging and visualization tools")
    print("• Integration with existing delayed generalization framework")

def show_attribute_combinations():
    """Show popular attribute combinations for research"""
    
    print("\n📋 Popular Attribute Combinations for Research:")
    print("=" * 55)
    
    combinations = [
        ("Male", "Blond_Hair", "Gender bias in hair color stereotypes"),
        ("Young", "Heavy_Makeup", "Age bias in appearance expectations"),
        ("Attractive", "Eyeglasses", "Beauty standards vs intelligence stereotypes"),
        ("Male", "Wearing_Lipstick", "Gender expression and cosmetics"),
        ("Young", "Eyeglasses", "Age stereotypes about vision/intelligence"),
        ("Attractive", "Heavy_Makeup", "Natural beauty vs enhanced appearance"),
        ("Male", "Heavy_Makeup", "Gender norms in cosmetics use"),
        ("Smiling", "Attractive", "Expression bias in attractiveness rating")
    ]
    
    for i, (attr1, attr2, description) in enumerate(combinations, 1):
        print(f"{i:2d}. {attr1:12} vs {attr2:15} - {description}")
    
    print(f"\nEach combination enables studying different aspects of bias:")
    print("• Social stereotypes and their algorithmic manifestation")
    print("• Fairness across protected demographic characteristics")
    print("• Delayed generalization from spurious to true correlations")
    print("• Real-world relevance to face recognition bias problems")

if __name__ == "__main__":
    demo_usage()
    show_attribute_combinations()
    
    print(f"\n🚀 Ready to run real CelebA bias experiments!")
    print(f"See GETTING_STARTED.md and data/vision/celeba/README.md for detailed guides.")