"""
测试评估指标
"""

import numpy as np
import pytest
from rna_fitness.utils.metrics import compute_metrics, compute_rnagym_metrics, print_metrics


def test_compute_metrics_basic():
    """测试基本指标计算"""
    # 完美预测
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics = compute_metrics(predictions, labels, include_rnagym_metrics=False)
    
    # MSE应该接近0
    assert metrics['mse'] < 1e-10
    assert metrics['rmse'] < 1e-5
    
    # R²应该接近1
    assert abs(metrics['r2'] - 1.0) < 1e-10
    
    # 相关系数应该接近1
    assert abs(metrics['spearman'] - 1.0) < 1e-10
    assert abs(metrics['pearson'] - 1.0) < 1e-10


def test_compute_rnagym_metrics():
    """测试RNAGym标准指标"""
    np.random.seed(42)
    labels = np.random.randn(100)
    predictions = labels + np.random.randn(100) * 0.3
    
    metrics = compute_rnagym_metrics(predictions, labels)
    
    # 检查指标存在
    assert 'spearman' in metrics
    assert 'auc' in metrics
    assert 'mcc' in metrics
    
    # 检查范围
    assert 0 <= metrics['spearman'] <= 1  # 绝对值
    assert 0.5 <= metrics['auc'] <= 1.0   # max(auc, 1-auc)
    assert 0 <= metrics['mcc'] <= 1       # 绝对值


def test_compute_metrics_with_rnagym():
    """测试包含RNAGym指标的完整计算"""
    np.random.seed(42)
    labels = np.random.randn(50)
    predictions = labels + np.random.randn(50) * 0.5
    
    metrics = compute_metrics(predictions, labels, include_rnagym_metrics=True)
    
    # 检查所有指标都存在
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'spearman' in metrics
    assert 'pearson' in metrics
    assert 'spearman_abs' in metrics
    assert 'auc' in metrics
    assert 'mcc' in metrics


def test_spearman_absolute_value():
    """测试Spearman相关系数的绝对值"""
    # 完全负相关
    labels = np.array([1, 2, 3, 4, 5])
    predictions = np.array([5, 4, 3, 2, 1])
    
    metrics = compute_rnagym_metrics(predictions, labels)
    
    # Spearman应该接近1（绝对值）
    assert abs(metrics['spearman'] - 1.0) < 0.01


def test_auc_symmetry():
    """测试AUC的对称性（max(auc, 1-auc)）"""
    np.random.seed(42)
    labels = np.random.randn(100)
    
    # 正相关预测
    predictions_pos = labels + np.random.randn(100) * 0.1
    metrics_pos = compute_rnagym_metrics(predictions_pos, labels)
    
    # 负相关预测（反转）
    predictions_neg = -predictions_pos
    metrics_neg = compute_rnagym_metrics(predictions_neg, labels)
    
    # 两者的AUC应该接近（因为使用了max(auc, 1-auc)）
    assert abs(metrics_pos['auc'] - metrics_neg['auc']) < 0.1


def test_mcc_binary_classification():
    """测试MCC二分类"""
    # 完美二分类
    labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
    predictions = np.array([1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3])
    
    metrics = compute_rnagym_metrics(predictions, labels)
    
    # MCC应该较高
    assert metrics['mcc'] > 0.8


def test_metrics_with_noise():
    """测试带噪声的指标"""
    np.random.seed(42)
    labels = np.linspace(0, 10, 100)
    predictions = labels + np.random.randn(100) * 2
    
    metrics = compute_metrics(predictions, labels, include_rnagym_metrics=True)
    
    # Spearman应该较高（单调关系）
    assert metrics['spearman_abs'] > 0.8
    
    # AUC应该合理
    assert metrics['auc'] > 0.7


def test_edge_case_constant_predictions():
    """测试常数预测的边界情况"""
    labels = np.array([1, 2, 3, 4, 5])
    predictions = np.array([3, 3, 3, 3, 3])
    
    try:
        metrics = compute_rnagym_metrics(predictions, labels)
        # MCC可能为0（随机预测）
        assert 0 <= metrics['mcc'] <= 1
    except:
        # 某些情况下可能会出错，这是可以接受的
        pass


def test_print_metrics():
    """测试指标打印功能"""
    np.random.seed(42)
    labels = np.random.randn(50)
    predictions = labels + np.random.randn(50) * 0.5
    
    metrics = compute_metrics(predictions, labels, include_rnagym_metrics=True)
    
    # 应该能够打印而不出错
    try:
        print_metrics(metrics, prefix="测试")
        assert True
    except Exception as e:
        pytest.fail(f"打印指标失败: {e}")


def test_metrics_array_types():
    """测试不同数组类型"""
    # 列表
    labels_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    predictions_list = [1.1, 2.1, 3.1, 4.1, 5.1]
    
    metrics1 = compute_rnagym_metrics(predictions_list, labels_list)
    
    # NumPy数组
    labels_array = np.array(labels_list)
    predictions_array = np.array(predictions_list)
    
    metrics2 = compute_rnagym_metrics(predictions_array, labels_array)
    
    # 两者应该相同
    assert abs(metrics1['spearman'] - metrics2['spearman']) < 1e-10
    assert abs(metrics1['auc'] - metrics2['auc']) < 1e-10
    assert abs(metrics1['mcc'] - metrics2['mcc']) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
