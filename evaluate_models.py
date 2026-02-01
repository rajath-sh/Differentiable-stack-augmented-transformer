"""
Evaluate Saved Models (No Training Required)
=============================================

This script loads pre-trained model checkpoints and runs evaluation
on Dyck-2 (multi-bracket) test sets. Use this after training is complete.

Usage:
    python evaluate_models.py

Requirements:
    - Saved model checkpoints in outputs/checkpoints/
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    MODEL_NAME, BATCH_SIZE, MAX_SEQ_LENGTH, GRAPHS_DIR, CHECKPOINTS_DIR,
    DYCK_TEST_SAMPLES
)
from src.data import generate_dyck2_data, StackDataset, collate_fn
from src.models import PretrainedBaseline, StackAugmentedPretrained
from src.evaluation import (
    plot_test_accuracy, plot_confusion_matrices,
    evaluate_with_metrics, save_metrics_report
)


# Test categories for Dyck-2 (by length) - Excluding Extra Long for cleaner results
DYCK_TEST_CATEGORIES = [
    ("Short (6-10)", 6, 10),
    ("Medium (12-18)", 12, 18),
    ("Long (20-30)", 20, 30),
    ("Very Long (32-44)", 32, 44),
    # ("Extra Long (46-60)", 46, 60),  # Excluded - both models at ~50%
]


def load_model(model_class, checkpoint_path, device):
    """Load a model from checkpoint."""
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Loaded: {checkpoint_path}")
    return model, checkpoint


def generate_test_datasets(tokenizer):
    """Generate Dyck-2 test datasets."""
    print("\n" + "="*70)
    print("GENERATING TEST DATASETS (Dyck-2: Multi-Bracket)")
    print("="*70)
    
    test_loaders = {}
    test_data = {}
    
    print("\n--- Dyck-2 Test Sets (by sequence length) ---")
    for name, min_len, max_len in DYCK_TEST_CATEGORIES:
        seqs, labels = generate_dyck2_data(
            DYCK_TEST_SAMPLES, (min_len, max_len)
        )
        dataset = StackDataset(seqs, labels, tokenizer, MAX_SEQ_LENGTH)
        test_loaders[name] = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )
        test_data[name] = (seqs, labels)
        print(f"  {name}: {len(seqs)} samples")
    
    return test_loaders, test_data


def evaluate_models(baseline_model, stack_model, test_loaders, device):
    """Run evaluation on all test sets."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    results = {'names': [], 'baseline': [], 'stack': [], 'baseline_f1': [], 'stack_f1': []}
    all_baseline_cms = []
    all_stack_cms = []
    metrics_dir = os.path.join(os.path.dirname(GRAPHS_DIR), "metrics")
    
    print(f"\n{'Test Set':<25} {'Baseline':<12} {'Stack':<12} {'Δ Acc':<10} {'Δ F1':<10}")
    print("-"*70)
    
    for name, loader in test_loaders.items():
        baseline_res = evaluate_with_metrics(baseline_model, loader, device, "Baseline")
        stack_res = evaluate_with_metrics(stack_model, loader, device, "Stack")
        
        b_acc = baseline_res['metrics']['accuracy']
        s_acc = stack_res['metrics']['accuracy']
        b_f1 = baseline_res['metrics']['f1_score']
        s_f1 = stack_res['metrics']['f1_score']
        
        results['names'].append(name)
        results['baseline'].append(b_acc)
        results['stack'].append(s_acc)
        results['baseline_f1'].append(b_f1)
        results['stack_f1'].append(s_f1)
        
        all_baseline_cms.append(baseline_res['confusion_matrix'])
        all_stack_cms.append(stack_res['confusion_matrix'])
        
        d_acc = s_acc - b_acc
        d_f1 = s_f1 - b_f1
        print(f"{name:<25} {b_acc:>6.1f}%     {s_acc:>6.1f}%     {d_acc:>+5.1f}%    {d_f1:>+5.1f}%")
        
        save_metrics_report(baseline_res, stack_res, name, metrics_dir)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    avg_b = np.mean(results['baseline'])
    avg_s = np.mean(results['stack'])
    avg_bf1 = np.mean(results['baseline_f1'])
    avg_sf1 = np.mean(results['stack_f1'])
    
    print(f"\nAccuracy:  Baseline: {avg_b:.1f}%  |  Stack: {avg_s:.1f}%  |  Improvement: +{avg_s-avg_b:.1f}%")
    print(f"F1 Score:  Baseline: {avg_bf1:.1f}%  |  Stack: {avg_sf1:.1f}%  |  Improvement: +{avg_sf1-avg_bf1:.1f}%")
    
    return results, all_baseline_cms, all_stack_cms


def export_test_data(test_data, save_dir=None):
    """Export test datasets to CSV."""
    import csv
    
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(GRAPHS_DIR), "test_datasets")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPORTING TEST DATASETS")
    print("="*70)
    
    test_file = os.path.join(save_dir, "dyck2_test_data.csv")
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'sequence', 'label', 'label_name'])
        for name, (seqs, labels) in test_data.items():
            for seq, label in zip(seqs, labels):
                label_name = 'valid' if label == 0 else 'invalid'
                writer.writerow([name, seq, label, label_name])
    print(f"  ✓ Dyck-2 test data: {test_file}")
    
    return save_dir


def generate_graphs(results, all_baseline_cms, all_stack_cms):
    """Generate evaluation graphs."""
    print("\n" + "="*70)
    print("GENERATING GRAPHS")
    print("="*70)
    
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Graph: Test accuracy
    plot_test_accuracy(
        results['names'],
        results['baseline'],
        results['stack'],
        save_path=os.path.join(GRAPHS_DIR, 'graph2_test_accuracy.png')
    )
    
    # Graph: Confusion matrices
    total_baseline_cm = np.array([[0, 0], [0, 0]])
    total_stack_cm = np.array([[0, 0], [0, 0]])
    for bcm, scm in zip(all_baseline_cms, all_stack_cms):
        total_baseline_cm += np.array(bcm)
        total_stack_cm += np.array(scm)
    
    plot_confusion_matrices(total_baseline_cm, total_stack_cm)


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*70)
    print("STACK-AUGMENTED TRANSFORMER EVALUATION")
    print("Task: Dyck-2 (Multi-Bracket) - Loading saved models")
    print("="*70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Check for saved models
    baseline_path = os.path.join(CHECKPOINTS_DIR, "baseline_model.pt")
    stack_path = os.path.join(CHECKPOINTS_DIR, "stack_augmented_model.pt")
    
    if not os.path.exists(baseline_path) or not os.path.exists(stack_path):
        print("\n❌ ERROR: Model checkpoints not found!")
        print(f"   Expected: {baseline_path}")
        print(f"   Expected: {stack_path}")
        print("\n   Please run main.py first to train the models.")
        return
    
    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    baseline_model, _ = load_model(PretrainedBaseline, baseline_path, device)
    stack_model, _ = load_model(StackAugmentedPretrained, stack_path, device)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Generate test datasets
    test_loaders, test_data = generate_test_datasets(tokenizer)
    
    # Export test data
    export_test_data(test_data)
    
    # Run evaluation
    results, all_baseline_cms, all_stack_cms = evaluate_models(
        baseline_model, stack_model, test_loaders, device
    )
    
    # Generate graphs
    generate_graphs(results, all_baseline_cms, all_stack_cms)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"  Graphs saved to:      {GRAPHS_DIR}")
    print(f"  Test datasets saved:  outputs/test_datasets/")
    print(f"  Metrics saved:        outputs/metrics/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
