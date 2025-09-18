# Visualization Tools for Delayed Generalization Research

## 📋 Overview

This directory contains visualization tools specifically designed for studying delayed generalization phenomena. The tools help researchers understand training dynamics, phase transitions, attention patterns, and feature evolution during the delayed generalization process.

## 🛠️ Visualization Categories

### [Training Dynamics](./training_dynamics/)
- **Loss Curves**: Enhanced loss and accuracy plotting with phase transition detection
- **Generalization Gap**: Visualization of train-test performance gaps over time
- **Learning Rate Effects**: Interactive plots showing how learning rate schedules affect training
- **Batch Size Impact**: Visualizations of how batch size changes influence learning dynamics

### [Phase Transitions](./phase_transitions/)
- **Transition Detection**: Automatic detection and visualization of sudden performance jumps
- **Grokking Analysis**: Specialized plots for grokking phenomena in algorithmic tasks
- **Emergence Tracking**: Visualization of emergent abilities development
- **Sharpness Analysis**: Measuring and plotting the sharpness of phase transitions

### [Attention Analysis](./attention_analysis/)
- **Attention Evolution**: How attention patterns change during delayed generalization
- **Head Specialization**: Tracking specialization of different attention heads over time
- **Pattern Formation**: Visualization of attention pattern development
- **Circuit Analysis**: Understanding how attention circuits form during training

### [Feature Evolution](./feature_evolution/)
- **Representation Dynamics**: How learned representations evolve during training
- **Feature Emergence**: Tracking when different features become important
- **Bias Development**: Visualizing how spurious correlations are learned and overcome
- **Dimensionality Analysis**: PCA/t-SNE visualizations of feature space evolution

## 🚀 Quick Start

### Basic Training Dynamics Plot
```python
from visualization.training_dynamics import DelayedGeneralizationPlotter

# Create plotter
plotter = DelayedGeneralizationPlotter(
    phenomenon_type='grokking'  # or 'simplicity_bias', 'phase_transitions'
)

# Add training data
for epoch in range(num_epochs):
    plotter.add_epoch_data(
        epoch=epoch,
        train_loss=train_loss,
        test_loss=test_loss,
        train_acc=train_acc,
        test_acc=test_acc
    )

# Generate comprehensive plot
fig = plotter.create_training_dynamics_plot()
plt.show()

# Detect and highlight phase transitions
transitions = plotter.detect_phase_transitions()
plotter.add_transition_markers(transitions)
```

### Phase Transition Analysis
```python
from visualization.phase_transitions import PhaseTransitionAnalyzer

# Analyze grokking in algorithmic tasks
analyzer = PhaseTransitionAnalyzer(
    model_checkpoints=checkpoints,
    train_data=train_data,
    test_data=test_data
)

# Detect grokking
grokking_analysis = analyzer.analyze_grokking(
    transition_threshold=0.95,
    window_size=100
)

# Create grokking visualization
fig = analyzer.create_grokking_plot(grokking_analysis)
plt.show()
```

### Attention Pattern Visualization
```python
from visualization.attention_analysis import AttentionEvolutionPlotter

# Track attention evolution during training
attention_plotter = AttentionEvolutionPlotter(model=transformer_model)

# Collect attention data across epochs
for epoch in key_epochs:
    model.load_state_dict(checkpoints[epoch])
    attention_data = attention_plotter.extract_attention_patterns(
        dataloader=sample_loader,
        num_samples=100
    )
    attention_plotter.add_epoch_data(epoch, attention_data)

# Visualize attention evolution
fig = attention_plotter.create_attention_evolution_plot()
plt.show()
```

## 📊 Advanced Visualizations

### Interactive Dashboards
The visualization tools support creating interactive dashboards using Plotly for real-time monitoring of delayed generalization experiments.

### Integration with WandB
All visualizations can be automatically logged to Weights & Biases for experiment tracking and comparison.

### Custom Phenomenon Plotting
Each visualization tool can be customized for specific delayed generalization phenomena with appropriate metrics and analysis methods.

## 🎯 Best Practices

### Data Collection
- **Frequent Sampling**: Collect metrics every epoch or multiple times per epoch
- **Long Training**: Ensure sufficient training time to capture delayed phenomena
- **Multiple Seeds**: Visualize uncertainty across multiple random seeds
- **Checkpointing**: Save model checkpoints at key intervals for analysis

### Visualization Guidelines
- **Clear Transitions**: Mark phase transitions clearly with vertical lines or shading
- **Multiple Metrics**: Show complementary metrics (loss, accuracy, norms, etc.)
- **Statistical Confidence**: Include error bars or confidence intervals when appropriate
- **Interpretable Scales**: Use appropriate scales and units for different phenomena

## 🔗 Integration Examples

See individual directories for detailed documentation and examples:

- **[Training Dynamics](./training_dynamics/README.md)**: Comprehensive training visualization tools
- **[Phase Transitions](./phase_transitions/README.md)**: Specialized transition analysis
- **[Attention Analysis](./attention_analysis/README.md)**: Attention pattern visualization
- **[Feature Evolution](./feature_evolution/README.md)**: Representation dynamics tools

## 📚 References

1. Power et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
2. Nanda et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability" 
3. Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits"
4. Olsson et al. (2022). "In-context Learning and Induction Heads"