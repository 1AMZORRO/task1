"""
评估指标
参照RNAGym标准实现
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


def compute_metrics(predictions, labels):
    """
    计算RNA fitness预测的评估指标
    
    根据RNAGym标准，主要评估指标包括：
    - MSE (Mean Squared Error)
    - R² (R-squared)
    - Spearman correlation
    - Pearson correlation
    
    Args:
        predictions: 预测值数组
        labels: 真实标签数组
        
    Returns:
        包含各项指标的字典
    """
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    # 基本统计指标
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    # 相关系数
    spearman_corr, spearman_pval = spearmanr(predictions, labels)
    pearson_corr, pearson_pval = pearsonr(predictions, labels)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'spearman': float(spearman_corr),
        'spearman_pval': float(spearman_pval),
        'pearson': float(pearson_corr),
        'pearson_pval': float(pearson_pval),
    }
    
    return metrics


def print_metrics(metrics, prefix=""):
    """
    格式化打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 打印前缀
    """
    print(f"\n{prefix}评估结果:")
    print(f"  MSE:              {metrics['mse']:.6f}")
    print(f"  RMSE:             {metrics['rmse']:.6f}")
    print(f"  R²:               {metrics['r2']:.6f}")
    print(f"  Spearman相关系数:  {metrics['spearman']:.6f} (p={metrics['spearman_pval']:.4e})")
    print(f"  Pearson相关系数:   {metrics['pearson']:.6f} (p={metrics['pearson_pval']:.4e})")
