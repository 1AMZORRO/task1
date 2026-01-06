"""
评估指标计算模块 - 回归任务专用
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr


def calculate_spearman(y_true, y_pred):
    """
    计算Spearman相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        correlation: Spearman相关系数
        p_value: p值
    """
    correlation, p_value = spearmanr(y_true, y_pred)
    return correlation, p_value


def calculate_pearson(y_true, y_pred):
    """
    计算Pearson相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        correlation: Pearson相关系数
        p_value: p值
    """
    correlation, p_value = pearsonr(y_true, y_pred)
    return correlation, p_value


def calculate_r2_score(y_true, y_pred):
    """
    计算R² (决定系数)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        r2: R²分数
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return r2


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有回归评估指标
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        metrics: 包含所有指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Spearman相关系数 (秩相关)
    spearman_corr, spearman_p = calculate_spearman(y_true, y_pred)
    
    # Pearson相关系数 (线性相关)
    pearson_corr, pearson_p = calculate_pearson(y_true, y_pred)
    
    # R² 决定系数
    r2 = calculate_r2_score(y_true, y_pred)
    
    # MSE和RMSE (回归任务的基础指标)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE (平均绝对误差)
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'pearson': pearson_corr,
        'pearson_p': pearson_p,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics


def print_metrics(metrics, prefix=""):
    """
    打印回归评估指标
    
    Args:
        metrics: 指标字典
        prefix: 打印前缀
    """
    print(f"\n{prefix}评估指标:")
    print(f"  Spearman相关系数: {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"  Pearson相关系数: {metrics['pearson']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"  R²分数: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
