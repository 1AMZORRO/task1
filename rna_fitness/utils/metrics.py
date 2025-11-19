"""
评估指标
参照RNAGym标准实现
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, matthews_corrcoef
from scipy.stats import spearmanr, pearsonr


def compute_metrics(predictions, labels, include_rnagym_metrics=True):
    """
    计算RNA fitness预测的评估指标
    
    根据RNAGym标准，主要评估指标包括：
    - Spearman相关系数 (绝对值)
    - AUC (Area Under ROC Curve)
    - MCC (Matthews Correlation Coefficient，绝对值)
    
    额外提供的指标：
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - R² (R-squared)
    - Pearson相关系数
    
    Args:
        predictions: 预测值数组
        labels: 真实标签数组
        include_rnagym_metrics: 是否包含RNAGym标准指标（AUC, MCC）
        
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
    
    # RNAGym标准指标
    if include_rnagym_metrics:
        try:
            # Spearman绝对值（RNAGym使用绝对值）
            metrics['spearman_abs'] = float(abs(spearman_corr))
            
            # AUC: 使用中位数作为阈值进行二分类
            # 将fitness值转换为二分类（高于中位数=1，否则=0）
            median_label = np.median(labels)
            binary_labels = (labels > median_label).astype(int)
            
            # 计算AUC
            auc = roc_auc_score(y_true=binary_labels, y_score=predictions)
            # RNAGym使用max(auc, 1-auc)确保AUC >= 0.5
            metrics['auc'] = float(max(auc, 1 - auc))
            
            # MCC: Matthews相关系数（绝对值）
            # 将预测值也转换为二分类
            median_pred = np.median(predictions)
            binary_preds = (predictions > median_pred).astype(int)
            
            mcc = matthews_corrcoef(y_true=binary_labels, y_pred=binary_preds)
            metrics['mcc'] = float(abs(mcc))
            
        except Exception as e:
            print(f"警告: 无法计算RNAGym指标: {e}")
            metrics['spearman_abs'] = float(abs(spearman_corr))
            metrics['auc'] = np.nan
            metrics['mcc'] = np.nan
    
    return metrics


def compute_rnagym_metrics(predictions, labels):
    """
    仅计算RNAGym标准的三个指标
    
    Args:
        predictions: 预测值数组
        labels: 真实标签数组
        
    Returns:
        包含Spearman, AUC, MCC的字典
    """
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    # Spearman相关系数（绝对值）
    spearman_corr = spearmanr(predictions, labels).correlation
    
    # AUC
    median_label = np.median(labels)
    binary_labels = (labels > median_label).astype(int)
    auc = roc_auc_score(y_true=binary_labels, y_score=predictions)
    
    # MCC
    median_pred = np.median(predictions)
    binary_preds = (predictions > median_pred).astype(int)
    mcc = matthews_corrcoef(y_true=binary_labels, y_pred=binary_preds)
    
    return {
        'spearman': float(abs(spearman_corr)),
        'auc': float(max(auc, 1 - auc)),
        'mcc': float(abs(mcc))
    }


def print_metrics(metrics, prefix=""):
    """
    格式化打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 打印前缀
    """
    print(f"\n{prefix}评估结果:")
    
    # RNAGym标准指标（如果存在）
    if 'spearman_abs' in metrics or 'auc' in metrics or 'mcc' in metrics:
        print(f"\n  === RNAGym标准指标 ===")
        if 'spearman_abs' in metrics:
            print(f"  Spearman (绝对值):  {metrics['spearman_abs']:.6f}")
        if 'auc' in metrics:
            print(f"  AUC:                {metrics['auc']:.6f}")
        if 'mcc' in metrics:
            print(f"  MCC (绝对值):       {metrics['mcc']:.6f}")
    
    # 其他指标
    print(f"\n  === 其他指标 ===")
    if 'mse' in metrics:
        print(f"  MSE:                {metrics['mse']:.6f}")
    if 'rmse' in metrics:
        print(f"  RMSE:               {metrics['rmse']:.6f}")
    if 'r2' in metrics:
        print(f"  R²:                 {metrics['r2']:.6f}")
    if 'spearman' in metrics:
        spearman_str = f"{metrics['spearman']:.6f}"
        if 'spearman_pval' in metrics:
            spearman_str += f" (p={metrics['spearman_pval']:.4e})"
        print(f"  Spearman:           {spearman_str}")
    if 'pearson' in metrics:
        pearson_str = f"{metrics['pearson']:.6f}"
        if 'pearson_pval' in metrics:
            pearson_str += f" (p={metrics['pearson_pval']:.4e})"
        print(f"  Pearson:            {pearson_str}")
