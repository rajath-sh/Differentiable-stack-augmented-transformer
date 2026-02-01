"""
Advanced metrics computation: Confusion Matrix, Precision, Recall, F1 Score.
"""

import torch
import numpy as np
import os
import json


def compute_predictions(model, loader, device):
    """
    Get all predictions and labels from a model on a dataset.
    
    Returns:
        predictions: numpy array of predicted labels
        labels: numpy array of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_confusion_matrix(predictions, labels):
    """
    Compute confusion matrix manually (no sklearn dependency).
    
    Returns:
        matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    # For binary classification: 0 = valid/correct, 1 = invalid/incorrect
    tn = np.sum((predictions == 0) & (labels == 0))  # True Negative
    fp = np.sum((predictions == 1) & (labels == 0))  # False Positive
    fn = np.sum((predictions == 0) & (labels == 1))  # False Negative
    tp = np.sum((predictions == 1) & (labels == 1))  # True Positive
    
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(confusion_matrix):
    """
    Compute Precision, Recall, F1 Score from confusion matrix.
    
    Args:
        confusion_matrix: 2x2 array [[TN, FP], [FN, TP]]
    
    Returns:
        dict with accuracy, precision, recall, f1_score
    """
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def evaluate_with_metrics(model, loader, device, model_name="Model"):
    """
    Full evaluation with all metrics.
    
    Returns:
        dict with all metrics and confusion matrix
    """
    predictions, labels = compute_predictions(model, loader, device)
    conf_matrix = compute_confusion_matrix(predictions, labels)
    metrics = compute_metrics(conf_matrix)
    
    print(f"\n  [{model_name}] Detailed Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"    Precision: {metrics['precision']:.1f}%")
    print(f"    Recall:    {metrics['recall']:.1f}%")
    print(f"    F1 Score:  {metrics['f1_score']:.1f}%")
    
    return {
        'confusion_matrix': conf_matrix.tolist(),
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels
    }


def save_metrics_report(baseline_results, stack_results, test_name, save_dir):
    """
    Save a detailed metrics comparison report.
    """
    report = {
        'test_set': test_name,
        'baseline': {
            'confusion_matrix': baseline_results['confusion_matrix'],
            **baseline_results['metrics']
        },
        'stack_augmented': {
            'confusion_matrix': stack_results['confusion_matrix'],
            **stack_results['metrics']
        },
        'comparison': {
            'accuracy_improvement': stack_results['metrics']['accuracy'] - baseline_results['metrics']['accuracy'],
            'f1_improvement': stack_results['metrics']['f1_score'] - baseline_results['metrics']['f1_score'],
            'precision_improvement': stack_results['metrics']['precision'] - baseline_results['metrics']['precision'],
            'recall_improvement': stack_results['metrics']['recall'] - baseline_results['metrics']['recall'],
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"metrics_{test_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
