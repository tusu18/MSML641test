import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_model(model, x_test, y_test):

    # Predictions
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro')
    }
    
    return metrics

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_to_csv(metrics_list, filepath):
    """Save metrics to CSV file"""
    df = pd.DataFrame(metrics_list)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")

def plot_comparison(df, save_dir):
    """Create comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy vs Sequence Length
    if 'seq_length' in df.columns:
        plt.figure(figsize=(10, 6))
        for arch in df['architecture'].unique():
            subset = df[df['architecture'] == arch]
            plt.plot(subset['seq_length'], subset['accuracy'], 
                    marker='o', label=arch.upper())
        plt.xlabel('Sequence Length')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Sequence Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/accuracy_vs_seqlen.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # F1 Score comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.nsmallest(20, 'f1_score')
    plt.barh(range(len(df_sorted)), df_sorted['f1_score'])
    plt.yticks(range(len(df_sorted)), 
              [f"{row['architecture']}-{row['activation']}-{row['optimizer']}" 
               for _, row in df_sorted.iterrows()], fontsize=8)
    plt.xlabel('F1 Score')
    plt.title('Model Comparison by F1 Score')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Evaluation utilities loaded successfully")
