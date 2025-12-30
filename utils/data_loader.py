"""
RNA序列数据加载和预处理模块
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RNATokenizer:
    """RNA序列分词器"""
    
    def __init__(self):
        # RNA碱基词汇表: A, U, G, C, N(未知), PAD, CLS, SEP
        self.vocab = {
            'PAD': 0,
            'CLS': 1,
            'SEP': 2,
            'A': 3,
            'U': 4,
            'G': 5,
            'C': 6,
            'N': 7  # 未知碱基
        }
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
    def encode(self, sequence, max_length=512):
        """将RNA序列编码为token IDs"""
        # 添加CLS token
        tokens = [self.vocab['CLS']]
        
        # 编码序列
        for base in sequence.upper():
            if base in self.vocab:
                tokens.append(self.vocab[base])
            else:
                tokens.append(self.vocab['N'])
        
        # 添加SEP token
        tokens.append(self.vocab['SEP'])
        
        # 截断或填充
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += [self.vocab['PAD']] * (max_length - len(tokens))
            
        return tokens
    
    def decode(self, token_ids):
        """将token IDs解码回序列"""
        sequence = []
        for idx in token_ids:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                if token not in ['PAD', 'CLS', 'SEP']:
                    sequence.append(token)
        return ''.join(sequence)


class RNAGymDataset(Dataset):
    """RNAGym数据集"""
    
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Args:
            csv_file: CSV文件路径
            tokenizer: RNA分词器
            max_length: 最大序列长度
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 检查数据
        print(f"加载数据集: {csv_file}")
        print(f"样本数量: {len(self.data)}")
        print(f"序列长度范围: {self.data['sequence'].str.len().min()} - {self.data['sequence'].str.len().max()}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        fitness = row['DMS_score']
        
        # 编码序列
        input_ids = self.tokenizer.encode(sequence, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'fitness': torch.tensor(fitness, dtype=torch.float32)
        }


def load_rnagym_data(data_dir, dataset_name, tokenizer, batch_size=32, 
                     train_ratio=0.8, max_length=512, num_workers=0):
    """
    加载RNAGym数据集并划分训练集和验证集
    
    Args:
        data_dir: 数据目录路径
        dataset_name: 数据集名称（CSV文件名，不含扩展名）
        tokenizer: RNA分词器
        batch_size: 批次大小
        train_ratio: 训练集比例
        max_length: 最大序列长度
        num_workers: 数据加载器worker数量
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    csv_file = os.path.join(data_dir, f"{dataset_name}.csv")
    
    # 创建数据集
    full_dataset = RNAGymDataset(csv_file, tokenizer, max_length)
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_available_datasets(data_dir):
    """获取可用的数据集列表"""
    datasets = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            datasets.append(file[:-4])  # 移除.csv扩展名
    return sorted(datasets)
