"""
Metrics and evaluation utilities
"""

import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(predictions, labels, task_name='emotion'):
    """Compute classification metrics"""
    # Remove padding (-1 labels) if present
    mask = labels != -1
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Generate detailed report
    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    
    return {
        f'{task_name}_accuracy': accuracy,
        f'{task_name}_precision': precision,
        f'{task_name}_recall': recall,
        f'{task_name}_macro_f1': f1,
        f'{task_name}_weighted_f1': weighted_f1,
        f'{task_name}_report': report
    }


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', 
                         figsize=(10, 8), save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm


def generate_classification_report(y_true, y_pred, labels, target_names=None,
                                 output_dict=False):
    """Generate detailed classification report"""
    if target_names is None:
        target_names = [str(label) for label in labels]
    
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )
    
    return report


# TensorFlow Metrics
class MacroF1ScoreTF(tf.keras.metrics.Metric):
    """Macro F1 Score metric for TensorFlow"""
    
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super(MacroF1ScoreTF, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision_metrics = [
            tf.keras.metrics.Precision(name=f'precision_{i}')
            for i in range(num_classes)
        ]
        self.recall_metrics = [
            tf.keras.metrics.Recall(name=f'recall_{i}')
            for i in range(num_classes)
        ]
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Update precision and recall for each class
        for i in range(self.num_classes):
            y_true_binary = tf.cast(tf.equal(y_true, i), tf.float32)
            y_pred_binary = tf.cast(tf.equal(y_pred, i), tf.float32)
            
            self.precision_metrics[i].update_state(
                y_true_binary, y_pred_binary, sample_weight
            )
            self.recall_metrics[i].update_state(
                y_true_binary, y_pred_binary, sample_weight
            )
    
    def result(self):
        f1_scores = []
        
        for i in range(self.num_classes):
            precision = self.precision_metrics[i].result()
            recall = self.recall_metrics[i].result()
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            f1_scores.append(f1)
        
        # Return macro average
        return tf.reduce_mean(f1_scores)
    
    def reset_state(self):
        for metric in self.precision_metrics:
            metric.reset_state()
        for metric in self.recall_metrics:
            metric.reset_state()


# PyTorch Metrics
class MacroF1ScorePT:
    """Macro F1 Score calculator for PyTorch"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.true_positives = torch.zeros(self.num_classes)
        self.false_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
    
    def update(self, predictions, targets):
        """Update metrics with batch predictions"""
        # Convert logits to predictions if needed
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=-1)
        
        # Calculate per-class statistics
        for i in range(self.num_classes):
            pred_i = predictions == i
            true_i = targets == i
            
            self.true_positives[i] += (pred_i & true_i).sum().float()
            self.false_positives[i] += (pred_i & ~true_i).sum().float()
            self.false_negatives[i] += (~pred_i & true_i).sum().float()
    
    def compute(self):
        """Compute macro F1 score"""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return f1_scores.mean().item()


def calculate_class_weights(labels, num_classes):
    """Calculate class weights for imbalanced datasets"""
    # Count occurrences
    counts = np.bincount(labels, minlength=num_classes)
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    weights = total / (num_classes * counts + 1e-7)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights


def plot_metrics_comparison(metrics_dict, metric_names=['accuracy', 'macro_f1'],
                          title='Model Performance Comparison', save_path=None):
    """Plot comparison of metrics across models"""
    models = list(metrics_dict.keys())
    num_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(f'emotion_{metric}', 0) for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Unified metric interface
def MacroF1Score(framework='tensorflow', num_classes=7):
    """Create macro F1 score metric for specified framework"""
    if framework == 'tensorflow':
        return MacroF1ScoreTF(num_classes)
    else:
        return MacroF1ScorePT(num_classes)