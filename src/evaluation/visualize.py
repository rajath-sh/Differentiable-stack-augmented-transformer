"""
Visualization and graph generation for Stack-Augmented Transformer results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ..config import GRAPHS_DIR


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')


def plot_confusion_matrices(baseline_cm, stack_cm, save_path=None):
    """
    Plot Graph 3: Side-by-side Confusion Matrices for Baseline and Stack-Augmented.
    
    Args:
        baseline_cm: 2x2 confusion matrix for baseline [[TN, FP], [FN, TP]]
        stack_cm: 2x2 confusion matrix for stack-augmented
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Valid\n(Predicted)', 'Invalid\n(Predicted)']
    true_labels = ['Valid (Actual)', 'Invalid (Actual)']
    
    # Baseline confusion matrix
    ax1 = axes[0]
    im1 = ax1.imshow(baseline_cm, cmap='Reds', alpha=0.8)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_yticklabels(true_labels, fontsize=11)
    ax1.set_title('Baseline Transformer\nConfusion Matrix', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if baseline_cm[i][j] > baseline_cm.max()/2 else 'black'
            ax1.text(j, i, f'{baseline_cm[i][j]}',
                    ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    
    # Stack confusion matrix
    ax2 = axes[1]
    im2 = ax2.imshow(stack_cm, cmap='Greens', alpha=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(true_labels, fontsize=11)
    ax2.set_title('Stack-Augmented Transformer\nConfusion Matrix', fontsize=13, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if stack_cm[i][j] > stack_cm.max()/2 else 'black'
            ax2.text(j, i, f'{stack_cm[i][j]}',
                    ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, 'graph3_confusion_matrices.png')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph 3 saved: {save_path}")
    plt.close()
    
    return fig


def plot_training_curves(baseline_losses, stack_losses, baseline_val_accs=None, stack_val_accs=None,
                         save_path=None):
    """
    Plot Graph 1: Training Loss over epochs.
    
    Args:
        baseline_losses: List of training losses for baseline model
        stack_losses: List of training losses for stack-augmented model
        baseline_val_accs: (unused, kept for backward compatibility)
        stack_val_accs: (unused, kept for backward compatibility)
        save_path: Path to save the figure (optional)
    
    Returns:
        fig: The matplotlib figure
    """
    epochs = range(1, len(baseline_losses) + 1)
    
    # Single plot for training loss only
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors
    baseline_color = '#d62728'  # Red
    stack_color = '#2ca02c'     # Green
    
    # Plot Training Loss
    ax.plot(epochs, baseline_losses, 'o-', label='Baseline Transformer',
            color=baseline_color, linewidth=2.5, markersize=6, alpha=0.9)
    ax.plot(epochs, stack_losses, 's-', label='Stack-Augmented Transformer',
            color=stack_color, linewidth=2.5, markersize=6, alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, len(epochs) + 0.5])
    
    # Add final loss annotations
    final_baseline = baseline_losses[-1]
    final_stack = stack_losses[-1]
    ax.annotate(f'Final: {final_baseline:.3f}', 
                xy=(len(epochs), final_baseline), 
                xytext=(len(epochs)-3, final_baseline+0.02),
                fontsize=10, color=baseline_color, fontweight='bold')
    ax.annotate(f'Final: {final_stack:.3f}', 
                xy=(len(epochs), final_stack), 
                xytext=(len(epochs)-3, final_stack-0.02),
                fontsize=10, color=stack_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, 'graph1_training_curves.png')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph 1 saved: {save_path}")
    
    return fig


def plot_test_accuracy(test_names, baseline_scores, stack_scores, save_path=None):
    """
    Plot Graph 2: Test Accuracy comparison across different sequence lengths.
    
    Args:
        test_names: List of test set names (e.g., ["Short", "Medium", "Long"])
        baseline_scores: List of baseline accuracies
        stack_scores: List of stack-augmented accuracies
        save_path: Path to save the figure (optional)
    
    Returns:
        fig: The matplotlib figure
    """
    x = np.arange(len(test_names))
    width = 0.35
    
    # Colors
    baseline_color = '#d62728'  # Red
    stack_color = '#2ca02c'     # Green
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline Transformer',
                   color=baseline_color, alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, stack_scores, width, label='Stack-Augmented Transformer',
                   color=stack_color, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement arrows/annotations
    for i, (b_score, s_score) in enumerate(zip(baseline_scores, stack_scores)):
        improvement = s_score - b_score
        if improvement > 0:
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(x[i] + width/2, s_score + 5),
                       fontsize=9, fontweight='bold', color='darkgreen',
                       ha='center')
    
    # Labels and formatting
    ax.set_xlabel('Test Sequence Length Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Stack-Augmented Transformer vs Baseline: Test Accuracy by Sequence Length',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, fontsize=11)
    ax.set_ylim([0, 115])
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Average improvement annotation
    avg_baseline = np.mean(baseline_scores)
    avg_stack = np.mean(stack_scores)
    avg_improvement = avg_stack - avg_baseline
    
    improvement_text = f'Average Improvement: +{avg_improvement:.1f}%'
    ax.text(0.02, 0.02, improvement_text,
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.95,
                     edgecolor='darkgreen', linewidth=2),
            ha='left', va='bottom')
    
    # Additional stats
    stats_text = f'Baseline Avg: {avg_baseline:.1f}%  |  Stack Avg: {avg_stack:.1f}%'
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9,
                     edgecolor='gray', linewidth=1),
            ha='right', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(GRAPHS_DIR, 'graph2_test_accuracy.png')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph 2 saved: {save_path}")
    
    return fig


def plot_combined_summary(baseline_history, stack_history, test_results, save_dir=None):
    """
    Generate both graphs and save them.
    
    Args:
        baseline_history: Dict with 'train_losses' and 'val_accuracies'
        stack_history: Dict with 'train_losses' and 'val_accuracies'
        test_results: Dict with 'names', 'baseline_scores', 'stack_scores'
        save_dir: Directory to save figures (optional)
    """
    if save_dir is None:
        save_dir = GRAPHS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Graph 1: Training curves
    plot_training_curves(
        baseline_history['train_losses'],
        stack_history['train_losses'],
        baseline_history['val_accuracies'],
        stack_history['val_accuracies'],
        save_path=os.path.join(save_dir, 'graph1_training_curves.png')
    )
    
    # Graph 2: Test accuracy
    plot_test_accuracy(
        test_results['names'],
        test_results['baseline_scores'],
        test_results['stack_scores'],
        save_path=os.path.join(save_dir, 'graph2_test_accuracy.png')
    )
    
    print(f"\n✓ All graphs saved to: {save_dir}")


def plot_additional_metrics(baseline_history, stack_history, test_results, save_dir=None):
    """
    Generate additional metric visualizations.
    
    Creates:
    - Graph 3: Improvement by sequence length (focused view)
    - Graph 4: Final comparison summary
    """
    if save_dir is None:
        save_dir = GRAPHS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== GRAPH 3: Improvement Analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    test_names = test_results['names']
    baseline_scores = test_results['baseline_scores']
    stack_scores = test_results['stack_scores']
    improvements = [s - b for s, b in zip(stack_scores, baseline_scores)]
    
    # Left: Improvement bar chart
    ax1 = axes[0]
    colors = ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements]
    bars = ax1.bar(range(len(test_names)), improvements, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels([n.split()[0] for n in test_names], fontsize=10)
    ax1.set_xlabel('Sequence Length Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Stack-Augmented Improvement Over Baseline', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Line plot comparison
    ax2 = axes[1]
    x = range(len(test_names))
    ax2.plot(x, baseline_scores, 'o-', label='Baseline', color='#d62728', 
             linewidth=2.5, markersize=10)
    ax2.plot(x, stack_scores, 's-', label='Stack-Augmented', color='#2ca02c',
             linewidth=2.5, markersize=10)
    
    # Fill between to show improvement
    ax2.fill_between(x, baseline_scores, stack_scores, 
                     where=[s >= b for s, b in zip(stack_scores, baseline_scores)],
                     color='#2ca02c', alpha=0.2, label='Improvement')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.split()[0] for n in test_names], fontsize=10)
    ax2.set_xlabel('Sequence Length Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Sequence Length', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 105])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'graph3_improvement_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph 3 saved: {save_path}")
    plt.close()
    
    # ===== GRAPH 4: Summary Dashboard =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Training loss comparison (final epochs)
    ax1 = axes[0, 0]
    epochs = range(1, len(baseline_history['train_losses']) + 1)
    ax1.plot(epochs, baseline_history['train_losses'], 'o-', label='Baseline',
             color='#d62728', linewidth=2, markersize=4)
    ax1.plot(epochs, stack_history['train_losses'], 's-', label='Stack-Augmented',
             color='#2ca02c', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Validation accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, baseline_history['val_accuracies'], 'o-', label='Baseline',
             color='#d62728', linewidth=2, markersize=4)
    ax2.plot(epochs, stack_history['val_accuracies'], 's-', label='Stack-Augmented',
             color='#2ca02c', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Accuracy Over Epochs', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Test accuracy grouped bar
    ax3 = axes[1, 0]
    x = np.arange(len(test_names))
    width = 0.35
    bars1 = ax3.bar(x - width/2, baseline_scores, width, label='Baseline',
                    color='#d62728', alpha=0.85)
    bars2 = ax3.bar(x + width/2, stack_scores, width, label='Stack-Augmented',
                    color='#2ca02c', alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.split()[0] for n in test_names], fontsize=9)
    ax3.set_xlabel('Test Category', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Test Accuracy by Sequence Length', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom-right: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    avg_baseline = np.mean(baseline_scores)
    avg_stack = np.mean(stack_scores)
    avg_improvement = avg_stack - avg_baseline
    
    summary_text = f"""
    ╔══════════════════════════════════════════╗
    ║         RESULTS SUMMARY                  ║
    ╠══════════════════════════════════════════╣
    ║  Baseline Average:     {avg_baseline:>6.1f}%           ║
    ║  Stack-Augmented Avg:  {avg_stack:>6.1f}%           ║
    ║  Average Improvement:  +{avg_improvement:>5.1f}%           ║
    ╠══════════════════════════════════════════╣
    ║  Best Improvement:                       ║
    ║  {test_names[improvements.index(max(improvements))]:20} +{max(improvements):.1f}%  ║
    ╠══════════════════════════════════════════╣
    ║  Final Training Loss:                    ║
    ║    Baseline:     {baseline_history['train_losses'][-1]:.4f}               ║
    ║    Stack:        {stack_history['train_losses'][-1]:.4f}               ║
    ╠══════════════════════════════════════════╣
    ║  Total Epochs: {len(epochs):3}                        ║
    ╚══════════════════════════════════════════╝
    """
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='center',
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'graph4_summary_dashboard.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Graph 4 saved: {save_path}")
    plt.close()
    
    return fig
