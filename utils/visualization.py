"""
Training visualization module
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                         save_path='training_curves.png'):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of training metrics dictionaries (optional)
        val_metrics: List of validation metrics dictionaries (optional)
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress Monitoring', fontsize=16)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'b-o', label='Training Loss', markersize=4)
    ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if train_metrics and val_metrics:
        # Spearman correlation
        ax = axes[0, 1]
        train_spearman = [m['spearman'] for m in train_metrics]
        val_spearman = [m['spearman'] for m in val_metrics]
        ax.plot(epochs, train_spearman, 'b-o', label='Training', markersize=4)
        ax.plot(epochs, val_spearman, 'r-s', label='Validation', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Spearman Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pearson correlation
        ax = axes[0, 2]
        train_pearson = [m['pearson'] for m in train_metrics]
        val_pearson = [m['pearson'] for m in val_metrics]
        ax.plot(epochs, train_pearson, 'b-o', label='Training', markersize=4)
        ax.plot(epochs, val_pearson, 'r-s', label='Validation', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Pearson Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² Score
        ax = axes[1, 0]
        train_r2 = [m['r2'] for m in train_metrics]
        val_r2 = [m['r2'] for m in val_metrics]
        ax.plot(epochs, train_r2, 'b-o', label='Training', markersize=4)
        ax.plot(epochs, val_r2, 'r-s', label='Validation', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score (Coefficient of Determination)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE
        ax = axes[1, 1]
        train_rmse = [m['rmse'] for m in train_metrics]
        val_rmse = [m['rmse'] for m in val_metrics]
        ax.plot(epochs, train_rmse, 'b-o', label='Training', markersize=4)
        ax.plot(epochs, val_rmse, 'r-s', label='Validation', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Square Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE
        ax = axes[1, 2]
        train_mae = [m['mae'] for m in train_metrics]
        val_mae = [m['mae'] for m in val_metrics]
        ax.plot(epochs, train_mae, 'b-o', label='Training', markersize=4)
        ax.plot(epochs, val_mae, 'r-s', label='Validation', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # If no metrics data, hide other subplots
        for i in range(1, 6):
            row, col = divmod(i, 3)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_prediction_scatter(y_true, y_pred, metrics, save_path='prediction_scatter.png'):
    """
    Plot scatter plot of predictions vs true values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Ideal prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    ax.set_xlabel('True Fitness Value', fontsize=12)
    ax.set_ylabel('Predicted Fitness Value', fontsize=12)
    ax.set_title('Prediction Results Comparison', fontsize=14)
    
    # Add metrics text
    textstr = f"Spearman: {metrics['spearman']:.4f}\n"
    textstr += f"Pearson: {metrics['pearson']:.4f}\n"
    textstr += f"R²: {metrics['r2']:.4f}\n"
    textstr += f"RMSE: {metrics['rmse']:.4f}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction scatter plot saved to: {save_path}")
    plt.close()


def save_results_summary(metrics, save_path='results_summary.txt'):
    """
    Save results summary to text file
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the summary
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("RNA Fitness Prediction Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Core Evaluation Metrics:\n")
        f.write(f"  Spearman Correlation: {metrics['spearman']:.6f}\n")
        f.write(f"  Spearman p-value: {metrics['spearman_p']:.6e}\n")
        f.write(f"  Pearson Correlation: {metrics['pearson']:.6f}\n")
        f.write(f"  Pearson p-value: {metrics['pearson_p']:.6e}\n")
        f.write(f"  R² Score: {metrics['r2']:.6f}\n\n")
        
        f.write("Regression Error Metrics:\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"  MAE: {metrics['mae']:.6f}\n")
        f.write(f"  MSE: {metrics['mse']:.6f}\n")
    
    print(f"Results summary saved to: {save_path}")
