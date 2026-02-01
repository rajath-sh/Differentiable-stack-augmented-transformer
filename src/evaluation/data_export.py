"""
Data export utilities for saving datasets and training results.
"""

import os
import json
import csv
from datetime import datetime

from ..config import GRAPHS_DIR


def export_full_datasets(train_seqs, train_labels, val_seqs, val_labels, 
                         test_datasets, save_dir=None):
    """
    Export FULL datasets (Dyck-1 + Dyck-2 combined) to CSV files.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(GRAPHS_DIR), "datasets")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Detect task type from sequence
    def get_task_type(seq):
        if '[' in seq or ']' in seq:
            return 'Dyck-2'
        return 'Dyck-1'
    
    # Export training data
    train_file = os.path.join(save_dir, "training_data.csv")
    with open(train_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'label', 'label_name', 'task_type'])
        for seq, label in zip(train_seqs, train_labels):
            label_name = 'valid' if label == 0 else 'invalid'
            task_type = get_task_type(seq)
            writer.writerow([seq, label, label_name, task_type])
    print(f"  ✓ Training data ({len(train_seqs)} samples): {train_file}")
    
    # Export validation data
    val_file = os.path.join(save_dir, "validation_data.csv")
    with open(val_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'label', 'label_name', 'task_type'])
        for seq, label in zip(val_seqs, val_labels):
            label_name = 'valid' if label == 0 else 'invalid'
            task_type = get_task_type(seq)
            writer.writerow([seq, label, label_name, task_type])
    print(f"  ✓ Validation data ({len(val_seqs)} samples): {val_file}")
    
    # Export test data by category
    test_file = os.path.join(save_dir, "test_data.csv")
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'sequence', 'label', 'label_name', 'task_type'])
        for name, (seqs, labels) in test_datasets.items():
            task_type = 'Dyck-1' if name.startswith('D1-') else 'Dyck-2'
            for seq, label in zip(seqs, labels):
                label_name = 'valid' if label == 0 else 'invalid'
                writer.writerow([name, seq, label, label_name, task_type])
    
    total_test = sum(len(seqs) for seqs, _ in test_datasets.values())
    print(f"  ✓ Test data ({total_test} samples): {test_file}")
    
    # Count by task type
    dyck1_train = sum(1 for s in train_seqs if '[' not in s and ']' not in s)
    dyck2_train = len(train_seqs) - dyck1_train
    
    # Export summary
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tasks': ['Dyck-1 (Single Bracket)', 'Dyck-2 (Multi-Bracket)'],
        'training_data': {
            'total': len(train_seqs),
            'dyck1_count': dyck1_train,
            'dyck2_count': dyck2_train,
            'valid_count': sum(1 for l in train_labels if l == 0),
            'invalid_count': sum(1 for l in train_labels if l == 1),
        },
        'validation_data': {
            'total': len(val_seqs)
        },
        'test_data': {
            name: len(seqs) for name, (seqs, _) in test_datasets.items()
        },
        'label_explanation': {
            '0 (valid)': 'Correctly balanced brackets',
            '1 (invalid)': 'Unbalanced or mismatched brackets'
        },
        'task_explanation': {
            'Dyck-1': 'Uses only ( ) - tests depth tracking',
            'Dyck-2': 'Uses ( ) and [ ] - tests depth + type tracking'
        }
    }
    
    summary_file = os.path.join(save_dir, "dataset_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Dataset summary: {summary_file}")
    
    return save_dir


def export_results(baseline_history, stack_history, test_results, save_dir=None):
    """
    Export training results to JSON for documentation.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(GRAPHS_DIR), "results")
    
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tasks': 'Dyck-1 + Dyck-2 (Combined)',
        'baseline': {
            'train_losses': baseline_history['train_losses'],
            'val_accuracies': baseline_history['val_accuracies'],
            'final_val_accuracy': baseline_history['val_accuracies'][-1],
        },
        'stack_augmented': {
            'train_losses': stack_history['train_losses'],
            'val_accuracies': stack_history['val_accuracies'],
            'final_val_accuracy': stack_history['val_accuracies'][-1],
        },
        'test_results': {
            'categories': test_results['names'],
            'baseline_scores': test_results['baseline_scores'],
            'stack_scores': test_results['stack_scores'],
            'improvements': [s - b for s, b in zip(test_results['stack_scores'], 
                                                    test_results['baseline_scores'])],
        },
        'summary': {
            'avg_baseline': sum(test_results['baseline_scores']) / len(test_results['baseline_scores']),
            'avg_stack': sum(test_results['stack_scores']) / len(test_results['stack_scores']),
        }
    }
    results['summary']['avg_improvement'] = results['summary']['avg_stack'] - results['summary']['avg_baseline']
    
    results_file = os.path.join(save_dir, "training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Training results: {results_file}")
    
    return results_file
