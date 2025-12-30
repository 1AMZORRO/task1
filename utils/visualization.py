"""
训练过程可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(train_losses, val_losses, train_metrics=None, val_metrics=None, 
                         save_path='training_curves.png'):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标字典列表 (可选)
        val_metrics: 验证指标字典列表 (可选)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Process Monitoring', fontsize=16)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4)
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
        ax.plot(epochs, train_spearman, 'b-o', label='Train', markersize=4)
        ax.plot(epochs, val_spearman, 'r-s', label='Val', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Spearman Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AUC
        ax = axes[1, 0]
        train_auc = [m['auc'] for m in train_metrics]
        val_auc = [m['auc'] for m in val_metrics]
        ax.plot(epochs, train_auc, 'b-o', label='Train', markersize=4)
        ax.plot(epochs, val_auc, 'r-s', label='Val', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('AUC Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MCC
        ax = axes[1, 1]
        train_mcc = [m['mcc'] for m in train_metrics]
        val_mcc = [m['mcc'] for m in val_metrics]
        ax.plot(epochs, train_mcc, 'b-o', label='Train', markersize=4)
        ax.plot(epochs, val_mcc, 'r-s', label='Val', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MCC')
        ax.set_title('Matthews Correlation Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # If no metrics data, hide other subplots
        for i in range(1, 4):
            row, col = divmod(i, 2)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_prediction_scatter(y_true, y_pred, metrics, save_path='prediction_scatter.png'):
    """
    绘制预测值vs真实值散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 评估指标字典
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Ideal prediction line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    ax.set_xlabel('True Fitness', fontsize=12)
    ax.set_ylabel('Predicted Fitness', fontsize=12)
    ax.set_title('Prediction Results', fontsize=14)
    
    # 添加指标文本
    textstr = f"Spearman: {metrics['spearman']:.4f}\n"
    textstr += f"AUC: {metrics['auc']:.4f}\n"
    textstr += f"MCC: {metrics['mcc']:.4f}"
    
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
    保存结果摘要到文本文件
    
    Args:
        metrics: 评估指标字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("RNA Fitness Prediction Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Core Evaluation Metrics:\n")
        f.write(f"  Spearman Correlation: {metrics['spearman']:.6f}\n")
        f.write(f"  Spearman p-value: {metrics['spearman_p']:.6e}\n")
        f.write(f"  AUC Score: {metrics['auc']:.6f}\n")
        f.write(f"  MCC Score: {metrics['mcc']:.6f}\n")
    
    print(f"Results summary saved to: {save_path}")
