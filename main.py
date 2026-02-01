"""
Stack-Augmented Transformer Training
=====================================

Main entry point for training and evaluating the Stack-Augmented Transformer
model compared to a Baseline Transformer.

Tasks:
- Dyck-1: Single bracket type ( )
- Dyck-2: Multi-bracket types ( ) and [ ]

Usage:
    python main.py
"""

import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports
from src.config import (
    MODEL_NAME, RANDOM_SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MAX_SEQ_LENGTH, GRAPHS_DIR, CHECKPOINTS_DIR,
    DYCK_VAL_LENGTH_RANGE, TEST_LENGTH_CATEGORIES
)
from src.data import generate_dyck_data, generate_dyck2_data, StackDataset, collate_fn
from src.models import PretrainedBaseline, StackAugmentedPretrained
from src.training import Trainer
from src.evaluation import (
    plot_training_curves, plot_test_accuracy, plot_confusion_matrices,
    export_full_datasets, export_results,
    evaluate_with_metrics, save_metrics_report
)

# LARGER DATASET CONFIGURATION
DYCK1_TRAIN_SAMPLES = 5000  # Single bracket
DYCK2_TRAIN_SAMPLES = 5000  # Multi-bracket
DYCK1_VAL_SAMPLES = 1000
DYCK2_VAL_SAMPLES = 1000
TEST_SAMPLES_PER_CATEGORY = 500

DYCK1_TRAIN_LENGTH = (4, 16)
DYCK2_TRAIN_LENGTH = (4, 16)


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_datasets(tokenizer):
    """Generate combined training data from Dyck-1 and Dyck-2."""
    print("\n" + "="*70)
    print("GENERATING DATASETS (Dyck-1 + Dyck-2 Combined)")
    print("="*70)
    
    # Training data - combine Dyck-1 and Dyck-2
    print("\nGenerating training data...")
    
    # Dyck-1 (single bracket)
    dyck1_train, dyck1_labels = generate_dyck_data(
        DYCK1_TRAIN_SAMPLES, DYCK1_TRAIN_LENGTH
    )
    print(f"  Dyck-1 training: {len(dyck1_train)} samples")
    
    # Dyck-2 (multi-bracket)
    dyck2_train, dyck2_labels = generate_dyck2_data(
        DYCK2_TRAIN_SAMPLES, DYCK2_TRAIN_LENGTH
    )
    print(f"  Dyck-2 training: {len(dyck2_train)} samples")
    
    # Combine training data
    train_seqs = dyck1_train + dyck2_train
    train_labels = dyck1_labels + dyck2_labels
    
    combined = list(zip(train_seqs, train_labels))
    random.shuffle(combined)
    train_seqs, train_labels = zip(*combined)
    train_seqs, train_labels = list(train_seqs), list(train_labels)
    
    print(f"  Total training: {len(train_seqs)} samples")
    
    # Validation data - also combined
    print("Generating validation data...")
    dyck1_val, dyck1_val_labels = generate_dyck_data(
        DYCK1_VAL_SAMPLES, DYCK_VAL_LENGTH_RANGE
    )
    dyck2_val, dyck2_val_labels = generate_dyck2_data(
        DYCK2_VAL_SAMPLES, DYCK_VAL_LENGTH_RANGE
    )
    
    val_seqs = dyck1_val + dyck2_val
    val_labels = dyck1_val_labels + dyck2_val_labels
    
    combined = list(zip(val_seqs, val_labels))
    random.shuffle(combined)
    val_seqs, val_labels = zip(*combined)
    val_seqs, val_labels = list(val_seqs), list(val_labels)
    
    print(f"  Total validation: {len(val_seqs)} samples")
    
    # Test sets - SEPARATE for Dyck-1 and Dyck-2
    print("\nGenerating test sets...")
    test_datasets = {}
    
    # Dyck-1 test sets
    print("  --- Dyck-1 (Single Bracket) ---")
    for name, min_len, max_len in TEST_LENGTH_CATEGORIES:
        seqs, labels = generate_dyck_data(
            TEST_SAMPLES_PER_CATEGORY, (min_len, max_len)
        )
        test_name = f"D1-{name}"
        test_datasets[test_name] = (seqs, labels)
        print(f"    {test_name}: {len(seqs)} samples")
    
    # Dyck-2 test sets
    print("  --- Dyck-2 (Multi-Bracket) ---")
    for name, min_len, max_len in TEST_LENGTH_CATEGORIES:
        seqs, labels = generate_dyck2_data(
            TEST_SAMPLES_PER_CATEGORY, (min_len, max_len)
        )
        test_name = f"D2-{name}"
        test_datasets[test_name] = (seqs, labels)
        print(f"    {test_name}: {len(seqs)} samples")
    
    # Create PyTorch datasets
    train_dataset = StackDataset(train_seqs, train_labels, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = StackDataset(val_seqs, val_labels, tokenizer, MAX_SEQ_LENGTH)
    
    test_dataloaders = {}
    for name, (seqs, labels) in test_datasets.items():
        dataset = StackDataset(seqs, labels, tokenizer, MAX_SEQ_LENGTH)
        test_dataloaders[name] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )
    
    raw_data = {
        'train': (train_seqs, train_labels),
        'val': (val_seqs, val_labels),
        'test': test_datasets
    }
    
    return train_dataset, val_dataset, test_dataloaders, raw_data


def main():
    """Main training and evaluation pipeline."""
    print("\n" + "="*70)
    print("STACK-AUGMENTED TRANSFORMER TRAINING")
    print("Tasks: Dyck-1 (Single Bracket) + Dyck-2 (Multi-Bracket)")
    print("="*70)
    
    set_seeds(RANDOM_SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset, val_dataset, test_dataloaders, raw_data = generate_datasets(tokenizer)
    
    # Export datasets
    print("\n" + "="*70)
    print("EXPORTING DATASETS (before training)")
    print("="*70)
    
    train_seqs, train_labels = raw_data['train']
    val_seqs, val_labels = raw_data['val']
    test_datasets = raw_data['test']
    
    export_full_datasets(train_seqs, train_labels, val_seqs, val_labels, test_datasets)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"\nDataLoader batch size: {BATCH_SIZE}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize models
    print("\n" + "="*70)
    print("INITIALIZING MODELS")
    print("="*70)
    
    print("\nLoading Baseline Transformer...")
    baseline_model = PretrainedBaseline()
    
    print("Loading Stack-Augmented Transformer...")
    stack_model = StackAugmentedPretrained()
    
    # Train models
    print("\n" + "="*70)
    print("TRAINING (Encoder Frozen)")
    print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}")
    print("="*70)
    
    print("\n--- Training Baseline Transformer ---")
    baseline_trainer = Trainer(baseline_model, device, "Baseline")
    baseline_trainer.train(
        train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, freeze_encoder=True
    )
    baseline_history = baseline_trainer.get_history()
    
    print("\n--- Training Stack-Augmented Transformer ---")
    stack_trainer = Trainer(stack_model, device, "Stack")
    stack_trainer.train(
        train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, freeze_encoder=True
    )
    stack_history = stack_trainer.get_history()
    
    # Save checkpoints
    print("\n" + "="*70)
    print("SAVING MODEL CHECKPOINTS")
    print("="*70)
    
    baseline_trainer.save_model(os.path.join(CHECKPOINTS_DIR, "baseline_model.pt"))
    stack_trainer.save_model(os.path.join(CHECKPOINTS_DIR, "stack_augmented_model.pt"))
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    # Separate results for Dyck-1 and Dyck-2
    d1_names, d1_baseline, d1_stack = [], [], []
    d2_names, d2_baseline, d2_stack = [], [], []
    all_baseline_cms, all_stack_cms = [], []
    all_metrics = []
    
    metrics_dir = os.path.join(os.path.dirname(GRAPHS_DIR), "metrics")
    
    print("\n" + "-"*70)
    print("DYCK-1 (Single Bracket) RESULTS")
    print("-"*70)
    print(f"{'Test Set':<25} {'Baseline':<12} {'Stack':<12} {'Δ Acc':<10}")
    print("-"*70)
    
    for name, loader in test_dataloaders.items():
        if name.startswith("D1-"):
            baseline_results = evaluate_with_metrics(baseline_model, loader, device, "Baseline")
            stack_results = evaluate_with_metrics(stack_model, loader, device, "Stack")
            
            b_acc = baseline_results['metrics']['accuracy']
            s_acc = stack_results['metrics']['accuracy']
            
            d1_names.append(name.replace("D1-", ""))
            d1_baseline.append(b_acc)
            d1_stack.append(s_acc)
            all_baseline_cms.append(baseline_results['confusion_matrix'])
            all_stack_cms.append(stack_results['confusion_matrix'])
            
            d_acc = s_acc - b_acc
            print(f"{name:<25} {b_acc:>6.1f}%     {s_acc:>6.1f}%     {d_acc:>+5.1f}%")
            
            save_metrics_report(baseline_results, stack_results, name, metrics_dir)
            all_metrics.append({'name': name, 'baseline': baseline_results['metrics'], 'stack': stack_results['metrics']})
    
    d1_avg_b, d1_avg_s = np.mean(d1_baseline), np.mean(d1_stack)
    print(f"{'D1 AVERAGE':<25} {d1_avg_b:>6.1f}%     {d1_avg_s:>6.1f}%     {d1_avg_s-d1_avg_b:>+5.1f}%")
    
    print("\n" + "-"*70)
    print("DYCK-2 (Multi-Bracket) RESULTS")
    print("-"*70)
    print(f"{'Test Set':<25} {'Baseline':<12} {'Stack':<12} {'Δ Acc':<10}")
    print("-"*70)
    
    for name, loader in test_dataloaders.items():
        if name.startswith("D2-"):
            baseline_results = evaluate_with_metrics(baseline_model, loader, device, "Baseline")
            stack_results = evaluate_with_metrics(stack_model, loader, device, "Stack")
            
            b_acc = baseline_results['metrics']['accuracy']
            s_acc = stack_results['metrics']['accuracy']
            
            d2_names.append(name.replace("D2-", ""))
            d2_baseline.append(b_acc)
            d2_stack.append(s_acc)
            all_baseline_cms.append(baseline_results['confusion_matrix'])
            all_stack_cms.append(stack_results['confusion_matrix'])
            
            d_acc = s_acc - b_acc
            print(f"{name:<25} {b_acc:>6.1f}%     {s_acc:>6.1f}%     {d_acc:>+5.1f}%")
            
            save_metrics_report(baseline_results, stack_results, name, metrics_dir)
            all_metrics.append({'name': name, 'baseline': baseline_results['metrics'], 'stack': stack_results['metrics']})
    
    d2_avg_b, d2_avg_s = np.mean(d2_baseline), np.mean(d2_stack)
    print(f"{'D2 AVERAGE':<25} {d2_avg_b:>6.1f}%     {d2_avg_s:>6.1f}%     {d2_avg_s-d2_avg_b:>+5.1f}%")
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    all_baseline = d1_baseline + d2_baseline
    all_stack = d1_stack + d2_stack
    overall_b, overall_s = np.mean(all_baseline), np.mean(all_stack)
    print(f"Dyck-1 Improvement: +{d1_avg_s - d1_avg_b:.1f}%")
    print(f"Dyck-2 Improvement: +{d2_avg_s - d2_avg_b:.1f}%")
    print(f"Overall Improvement: +{overall_s - overall_b:.1f}%")
    
    # Combined test results for graphs
    test_names = d1_names  # Use Dyck-1 names for main graph
    baseline_scores = d1_baseline
    stack_scores = d1_stack
    
    test_results = {
        'names': test_names,
        'baseline_scores': baseline_scores,
        'stack_scores': stack_scores
    }
    
    # Generate graphs
    print("\n" + "="*70)
    print("GENERATING GRAPHS")
    print("="*70)
    
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Graph 1: Training curves
    plot_training_curves(
        baseline_history['train_losses'],
        stack_history['train_losses'],
        baseline_history['val_accuracies'],
        stack_history['val_accuracies']
    )
    
    # Graph 2: Dyck-1 Test accuracy
    plot_test_accuracy(d1_names, d1_baseline, d1_stack,
                       save_path=os.path.join(GRAPHS_DIR, 'graph2_dyck1_test_accuracy.png'))
    
    # Graph 3: Dyck-2 Test accuracy
    plot_test_accuracy(d2_names, d2_baseline, d2_stack,
                       save_path=os.path.join(GRAPHS_DIR, 'graph3_dyck2_test_accuracy.png'))
    
    # Graph 4: Confusion matrices
    total_baseline_cm = np.array([[0, 0], [0, 0]])
    total_stack_cm = np.array([[0, 0], [0, 0]])
    for bcm, scm in zip(all_baseline_cms, all_stack_cms):
        total_baseline_cm += np.array(bcm)
        total_stack_cm += np.array(scm)
    
    plot_confusion_matrices(total_baseline_cm, total_stack_cm,
                           save_path=os.path.join(GRAPHS_DIR, 'graph4_confusion_matrices.png'))
    
    # Export results
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    export_results(baseline_history, stack_history, test_results)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"  Tasks: Dyck-1 + Dyck-2 (10,000 total training samples)")
    print(f"  Graphs:     {GRAPHS_DIR}")
    print(f"  Datasets:   outputs/datasets/")
    print(f"  Models:     {CHECKPOINTS_DIR}")
    print(f"  Metrics:    outputs/metrics/")
    print(f"\n  Dyck-1 Improvement: +{d1_avg_s - d1_avg_b:.1f}%")
    print(f"  Dyck-2 Improvement: +{d2_avg_s - d2_avg_b:.1f}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
