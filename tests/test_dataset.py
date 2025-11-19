"""
测试RNA数据集
"""

import torch
import pytest
import numpy as np
from rna_fitness.data.dataset import RNADataset
from rna_fitness.utils.tokenizer import RNATokenizer


def test_dataset_init_with_data():
    """测试用数据初始化数据集"""
    sequences = ['AUGC', 'UUUU', 'GGCC']
    fitness = [0.5, 0.8, 0.3]
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=20
    )
    
    assert len(dataset) == 3


def test_dataset_getitem():
    """测试获取数据集项"""
    sequences = ['AUGC', 'UUUU', 'GGCC']
    fitness = [0.5, 0.8, 0.3]
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=20
    )
    
    # 获取第一个样本
    sample = dataset[0]
    
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample
    
    assert isinstance(sample['input_ids'], torch.Tensor)
    assert isinstance(sample['attention_mask'], torch.Tensor)
    assert isinstance(sample['labels'], torch.Tensor)
    
    assert sample['input_ids'].shape == (20,)
    assert sample['attention_mask'].shape == (20,)
    assert sample['labels'].shape == ()


def test_dataset_label_values():
    """测试标签值"""
    sequences = ['AUGC', 'UUUU', 'GGCC']
    fitness = [0.5, 0.8, 0.3]
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=20
    )
    
    # 检查每个样本的标签
    for i in range(len(dataset)):
        sample = dataset[i]
        assert torch.isclose(sample['labels'], torch.tensor(fitness[i]))


def test_dataset_iteration():
    """测试数据集迭代"""
    sequences = ['AUGC'] * 10
    fitness = np.random.randn(10).tolist()
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=20
    )
    
    # 迭代所有样本
    count = 0
    for sample in dataset:
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        count += 1
    
    assert count == len(dataset)


def test_dataset_different_lengths():
    """测试不同长度的序列"""
    sequences = ['AUGC', 'AUGCAUGC', 'AUGCAUGCAUGC']
    fitness = [0.5, 0.8, 0.3]
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=30
    )
    
    # 所有样本应该被填充到相同长度
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample['input_ids'].shape == (30,)
        assert sample['attention_mask'].shape == (30,)


def test_dataset_attention_mask():
    """测试注意力mask"""
    sequences = ['AUGC', 'AUGCAUGCAUGC']
    fitness = [0.5, 0.8]
    tokenizer = RNATokenizer()
    
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=30
    )
    
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    # 第二个序列更长，应该有更多的非填充token
    assert sample1['attention_mask'].sum() < sample2['attention_mask'].sum()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
