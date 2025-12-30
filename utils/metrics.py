"""
评估指标计算模块
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize


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


def calculate_auc(y_true, y_pred, threshold=None):
    """
    计算AUC (Area Under ROC Curve)
    
    对于回归任务，需要将连续值转换为二分类问题
    使用中位数作为阈值
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 分类阈值，默认使用真实值的中位数
        
    Returns:
        auc: AUC分数
    """
    if threshold is None:
        threshold = np.median(y_true)
    
    # 转换为二分类标签
    y_true_binary = (y_true >= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true_binary, y_pred)
    except ValueError:
        # 如果只有一个类别，返回NaN
        auc = np.nan
        
    return auc


def calculate_mcc(y_true, y_pred, threshold=None):
    """
    计算Matthews相关系数 (MCC)
    
    对于回归任务，需要将连续值转换为二分类问题
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        threshold: 分类阈值，默认使用真实值的中位数
        
    Returns:
        mcc: MCC分数
    """
    if threshold is None:
        threshold = np.median(y_true)
    
    # 转换为二分类标签
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    try:
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
    except ValueError:
        # 如果只有一个类别，返回NaN
        mcc = np.nan
        
    return mcc


def calculate_all_metrics(y_true, y_pred):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        metrics: 包含所有指标的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Spearman相关系数
    spearman_corr, spearman_p = calculate_spearman(y_true, y_pred)
    
    # AUC
    auc = calculate_auc(y_true, y_pred)
    
    # MCC
    mcc = calculate_mcc(y_true, y_pred)
    
    # MSE和RMSE (回归任务的基础指标)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        'spearman': spearman_corr,
        'spearman_p': spearman_p,
        'auc': auc,
        'mcc': mcc,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics


def print_metrics(metrics, prefix=""):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 打印前缀
    """
    print(f"\n{prefix}评估指标:")
    print(f"  Spearman相关系数: {metrics['spearman']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
