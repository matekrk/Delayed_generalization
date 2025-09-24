# Opposing Signals Analysis - Unified Notebook

This directory contains tools for analyzing neural network training dynamics with a focus on detecting "opposing signals" as described in Rosenfeld & Risteski (2023): "Outliers with Opposing Signals Have an Outsized Effect on Neural Network Optimization".

## Files

### Main Notebook
- **`neural_network_training_dynamics_unified.ipynb`** - ✅ **WORKING VERSION** - Unified, improved implementation that combines and fixes issues from the previous versions

### Legacy Notebooks (for reference)
- `enhanced_neural_network_training_dynamics.ipynb` - Original working version
- `enhanced_neural_network_training_dynamics_with_animations.ipynb` - Version with animation issues

## Key Features of Unified Notebook

### ✅ Fixed Issues from Previous Versions
- **Improved opposing pairs detection** - Fixed algorithm that was struggling to detect opposing signals
- **Robust gradient computation** - Better handling of gradient calculations for individual examples
- **Enhanced animation system** - Configurable, high-quality animations with proper temporal sampling
- **Better statistical analysis** - More reliable thresholds and similarity metrics

### 🔧 Core Functionality
1. **Opposing Signals Detection**: Identifies example pairs with opposing gradient directions
2. **Learning Dynamics Tracking**: Monitor individual example learning/forgetting patterns  
3. **Loss Trajectory Analysis**: Track how individual examples' losses evolve over training
4. **Feature Learning Visualization**: Analyze how CNN features develop during training
5. **Class-wise Analysis**: Study how different classes exhibit different learning patterns

### 📊 Visualizations Generated
- Training progress curves (loss, accuracy, learning rate)
- Opposing signals detection over time  
- Example loss trajectories by class
- Confidence evolution patterns
- Gradient norms distribution
- Class pair matrix for opposing signals
- **Animated loss trajectories** (optional, configurable)

### 📈 Key Improvements
- **Configurable parameters** - Easy to adjust thresholds, sampling rates, animation settings
- **Performance optimized** - Efficient gradient computation and data collection
- **Better opposing detection** - Uses cosine similarity with magnitude checks
- **Comprehensive reporting** - Detailed analysis with actionable insights
- **Save/load functionality** - Preserve results for further analysis

## Usage

### Quick Start
```python
# 1. Configure parameters at the top of the notebook
DATA_PATH = './data'  # Your data directory
OPPOSING_THRESHOLD = 2.0  # Sensitivity for detecting opposing signals
SAVE_ANIMATIONS = True  # Set False for faster runs
N_EXAMPLES_TRACK = 100  # Number of examples to track in detail

# 2. Run all cells - the notebook will:
#    - Train a CNN on CIFAR-10
#    - Track training dynamics
#    - Detect opposing signals
#    - Generate visualizations
#    - Create comprehensive report
```

### Configuration Options

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `OPPOSING_THRESHOLD` | Gradient magnitude ratio threshold | 2.0 (normal), 1.5 (sensitive) |
| `N_EXAMPLES_TRACK` | Examples to track in detail | 100-200 |
| `DENSE_SAMPLING` | Collect data every epoch | True (better quality), False (faster) |
| `SAVE_ANIMATIONS` | Generate animated visualizations | True (full analysis), False (quick runs) |
| `ANIMATION_FPS` | Animation frame rate | 10 (smooth), 5 (smaller files) |

### Expected Outputs

The notebook generates:
- `training_dynamics_analysis.png` - Main analysis plots
- `opposing_signals_analysis.png` - Opposing signals visualization  
- `loss_trajectories.gif` - Animated trajectories (if enabled)
- `training_dynamics_report.txt` - Detailed text report
- `training_dynamics_results.pkl` - All data for further analysis

### Interpreting Results

#### Opposing Signals
- **High opposing signals**: Model learning conflicting patterns from different examples
- **Class pairs**: Shows which classes have most opposing examples
- **Evolution over time**: How opposing signals change during training

#### Learning Patterns
- **Loss trajectories**: Individual example difficulty progression
- **Confidence evolution**: How model certainty develops
- **Gradient norms**: Training signal strength for different examples

## Research Applications

This notebook is particularly useful for studying:

1. **Delayed Generalization**: How models initially memorize before generalizing
2. **Example Difficulty**: Which examples are hardest to learn
3. **Class Interactions**: How different classes interfere with each other
4. **Training Dynamics**: Understanding the learning process at example level
5. **Forgetting Patterns**: Which examples get forgotten during training

## Troubleshooting

### No Opposing Signals Detected
- Lower `OPPOSING_THRESHOLD` to 1.5 or 1.0
- Increase `N_EXAMPLES_TRACK` to 200+
- Enable `DENSE_SAMPLING` for more data points
- Train for more epochs (15+ recommended)

### Animation Issues
- Ensure you have sufficient examples with trajectories
- Check that `SAVE_ANIMATIONS=True`
- Verify matplotlib and pillow are installed
- Reduce `ANIMATION_FPS` if memory issues occur

### Performance Issues
- Reduce `N_EXAMPLES_TRACK` to 50-100
- Set `DENSE_SAMPLING=False`
- Set `SAVE_ANIMATIONS=False`
- Limit gradient computation to fewer examples

## Technical Details

### Opposing Signals Algorithm
1. Compute gradients for individual examples
2. Calculate cosine similarity between gradient vectors
3. Identify pairs with strong negative correlation (< -0.5)
4. Check gradient magnitude compatibility (within threshold ratio)
5. Filter by cross-class pairs (same class pairs excluded)

### Performance Optimizations
- Gradient computation limited to subset of batches
- Efficient tracking data structures
- Configurable sampling rates
- Memory-conscious animation generation

## Citation

If you use this notebook in your research, please cite:

```bibtex
@article{rosenfeld2023outliers,
  title={Outliers with Opposing Signals Have an Outsized Effect on Neural Network Optimization},
  author={Rosenfeld, Elan and Risteski, Andrej},
  year={2023}
}
```

## Dependencies

- PyTorch >= 1.8.0
- torchvision >= 0.9.0  
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- numpy >= 1.19.0
- pandas >= 1.2.0
- jupyter >= 1.0.0