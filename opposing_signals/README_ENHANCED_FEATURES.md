# Advanced Training Dynamics Analysis Features

This directory contains comprehensive analysis features that extend the capabilities of the `neural_network_training_dynamics_unified.ipynb` notebook with advanced training dynamics tracking and visualization.

## 🎯 Advanced Capabilities

The analysis features provide comprehensive capabilities for understanding training dynamics:

### ✅ **Complete Feature Set:**

1. **📈 Difficulty Over Time Tracking**
   - Track individual example difficulty progression
   - Visualize average difficulty trends across training
   - Class-wise difficulty evolution analysis

2. **📊 Separate Accuracy Evolution (Train/Test)**
   - Dedicated train vs test accuracy plots
   - Class-wise accuracy breakdown for both splits
   - Accuracy gap analysis per class

3. **🔄 Forgetting Events Analysis with Color-Coded Plots**
   - Track when examples are forgotten during training
   - Color-coded visualization by forgetting intensity
   - Timeline of forgetting events across epochs

4. **📈 Distribution of Forgetting Events**
   - Histogram of forgetting event frequencies
   - Statistical analysis of example memory patterns
   - Identification of "difficult" examples

5. **🎯 Class-wise Accuracy and Loss Evolution**
   - Individual class performance tracking
   - Wide-format plots for better visualization
   - Loss evolution per class over training

6. **⚡ Opposing Gradient Pairs Highlighting** 
   - Detection of examples with opposing gradient signals
   - Visualization of gradient similarity patterns
   - Highlighting during training with size indicators

7. **🚦 Loss Change Highlighting**
   - Green frames for positive loss changes (learning)
   - Red frames for negative loss changes (forgetting)
   - Dynamic highlighting of training instability

8. **🎬 Enhanced .gif Animations**
   - Bar plots for each epoch evaluation (not cumulative)
   - Class-wise accuracy bars
   - Training timeline with current epoch highlighting
   - Statistics panel with opposing pairs count

## 📁 Files

- **`enhanced_analysis_features.py`** - Main implementation with `EnhancedTrainingAnalyzer` class
- **`integration_example.py`** - Complete example showing how to use the analysis features
- **`README_ENHANCED_FEATURES.md`** - This documentation file

## 🚀 Quick Start

### Option 1: Standalone Usage

```python
from enhanced_analysis_features import EnhancedTrainingAnalyzer

# Initialize analyzer
analyzer = EnhancedTrainingAnalyzer(num_classes=10, max_examples_track=1000)

# In your training loop:
for epoch in range(num_epochs):
    # ... your training code ...
    
    # Track comprehensive dynamics
    analyzer.track_epoch(model, train_loader, test_loader, criterion, epoch, optimizer)
    
    # Generate plots periodically
    if epoch % 10 == 0:
        analyzer.plot_accuracy_evolution()
        analyzer.plot_difficulty_evolution()
        analyzer.plot_opposing_signals()

# Generate comprehensive report
analyzer.generate_comprehensive_report("./analysis_results")
```

### Option 2: Integration with Existing Unified Notebook

1. **Add imports to your notebook:**
```python
from enhanced_analysis_features import EnhancedTrainingAnalyzer
```

2. **Replace the existing tracker initialization:**
```python
# Use comprehensive tracking:
analyzer = EnhancedTrainingAnalyzer(num_classes=10, max_examples_track=1000)
```

3. **Update your training loop:**
```python
for epoch in range(num_epochs):
    # ... existing training code ...
    
    # Add comprehensive tracking:
    analyzer.track_epoch(model, train_loader, test_loader, criterion, epoch, optimizer)
```

4. **Generate comprehensive visualizations:**
```python
# All the comprehensive features are available:
analyzer.plot_difficulty_evolution()          # Difficulty over time
analyzer.plot_accuracy_evolution()            # Separate train/test with class breakdown  
analyzer.plot_opposing_signals()              # Opposing pairs + loss changes
analyzer.create_enhanced_animation("./anim.gif")  # Bar plot animations
```

## 🎨 Visualization Examples

### Difficulty Evolution Analysis
- **Average difficulty over time** - Shows how example difficulty changes during training
- **Class-wise difficulty trends** - Per-class difficulty progression
- **Forgetting events distribution** - Histogram of how often examples are forgotten
- **Color-coded forgetting timeline** - When forgetting events occur (red intensity = more events)

### Enhanced Accuracy Plots  
- **Train vs Test evolution** - Side-by-side accuracy progression with gap highlighting
- **Class-wise train accuracy** - Individual class performance during training
- **Class-wise test accuracy** - Individual class generalization patterns
- **Accuracy gap analysis** - Train-test gap per class over time

### Opposing Signals Analysis
- **Opposing pairs count** - Number of gradient-opposing example pairs per epoch
- **Similarity distribution** - Histogram of gradient cosine similarities
- **Loss changes with color coding** - Green bars (loss decrease) vs Red bars (loss increase)
- **Gradient norm distribution** - Magnitude of gradients for opposing examples

### Enhanced Animations
- **Bar plot format** - Each frame shows current epoch as bars (not cumulative lines)
- **Class-wise accuracy bars** - Dynamic bar chart of per-class performance
- **Training timeline** - Line plot showing progression with current epoch highlighted
- **Statistics panel** - Current metrics including opposing pairs count and loss changes

## 🔧 Advanced Configuration

### Analyzer Configuration
```python
analyzer = EnhancedTrainingAnalyzer(
    num_classes=10,                    # Number of classes in your dataset
    max_examples_track=1000            # Maximum examples to track individually
)
```

### Animation Customization
```python
anim = analyzer.create_enhanced_animation(
    save_path="./enhanced_training.gif",
    fps=10                             # Frames per second
)
```

### Comprehensive Report Generation
```python
report_dir = analyzer.generate_comprehensive_report(
    save_dir="./analysis_results"      # Output directory
)
# Generates:
#   - difficulty_evolution_<timestamp>.png
#   - accuracy_evolution_<timestamp>.png  
#   - opposing_signals_<timestamp>.png
#   - training_animation_<timestamp>.gif
#   - summary_statistics_<timestamp>.txt
```

## 📊 Output Files

The comprehensive analyzer generates detailed analysis outputs:

### Static Plots
- **Difficulty Evolution** - 4-panel plot with difficulty trends and forgetting analysis
- **Accuracy Evolution** - 4-panel plot with train/test and class-wise breakdowns
- **Opposing Signals** - 4-panel plot with opposing pairs and loss change analysis

### Animations
- **Enhanced Training GIF** - Multi-panel animation with:
  - Train vs Test accuracy bars
  - Class-wise accuracy bars  
  - Training timeline with current position
  - Statistics panel with key metrics

### Reports
- **Summary Statistics** - Text file with key training metrics and analysis results

## 🔍 Key Features Comparison

| Feature | Unified Notebook | Enhanced Features |
|---------|------------------|-------------------|
| Basic accuracy tracking | ✅ | ✅ |
| Difficulty over time | ❌ | ✅ |
| Separate train/test plots | ❌ | ✅ |
| Forgetting events analysis | ❌ | ✅ |
| Color-coded forgetting | ❌ | ✅ |
| Class-wise accuracy/loss | ❌ | ✅ |
| Opposing pairs detection | ✅ | ✅ Enhanced |
| Loss change highlighting | ❌ | ✅ |
| Bar plot animations | ❌ | ✅ |
| Comprehensive reports | ❌ | ✅ |

## 🎯 Use Cases

### Research Applications
- **Delayed Generalization Studies** - Track when and how generalization occurs
- **Example Difficulty Analysis** - Understand which examples drive learning
- **Class Imbalance Research** - Analyze per-class learning dynamics
- **Optimization Studies** - Visualize opposing gradient effects

### Educational Applications  
- **Teaching ML Concepts** - Visual demonstrations of training dynamics
- **Debugging Training** - Identify problematic examples or classes
- **Model Comparison** - Compare learning patterns across architectures

### Production Applications
- **Training Monitoring** - Real-time analysis of training health
- **Data Quality Assessment** - Identify problematic examples in datasets
- **Curriculum Learning** - Design better training schedules based on difficulty

## 🚀 Running the Example

```bash
# Navigate to the opposing_signals directory
cd opposing_signals/

# Run the integration example
python integration_example.py

# This will:
# 1. Train a CNN on CIFAR-10 subset
# 2. Track all comprehensive dynamics
# 3. Generate comprehensive visualizations
# 4. Create detailed animations
# 5. Save all results to ./analysis_results/
```

## 📝 Notes

- **Memory Usage**: The analyzer tracks individual examples, so memory usage scales with `max_examples_track`
- **Performance**: Gradient analysis for opposing signals adds computation overhead
- **Visualization**: All plots are high-resolution (300 DPI) suitable for publications
- **Compatibility**: Designed to work seamlessly with existing unified notebook code

## 🎉 Comprehensive Analysis

The complete feature set provides all the analysis capabilities needed for advanced training dynamics research while maintaining ease of use and integration with existing workflows!

The comprehensive features deliver the detailed analysis capabilities required for understanding training dynamics in delayed generalization research.