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
    
    def __init__(self, csv_files, tokenizer, max_length=512, normalize=True):
        """
        Args:
            csv_files: CSV文件路径或路径列表
            tokenizer: RNA分词器
            max_length: 最大序列长度
            normalize: 是否对fitness值进行标准化
        """
        # 支持单个文件或多个文件
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        
        # 加载并合并所有数据集
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            print(f"Loading dataset: {csv_file}")
            print(f"  Number of samples: {len(df)}")
            print(f"  Sequence length range: {df['sequence'].str.len().min()} - {df['sequence'].str.len().max()}")
            all_data.append(df)
        
        self.data = pd.concat(all_data, ignore_index=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.normalize = normalize
        
        # 对fitness值进行标准化（使用Z-score标准化）
        if self.normalize:
            # 计算均值和标准差
            self.fitness_mean = self.data['DMS_score'].mean()
            self.fitness_std = self.data['DMS_score'].std()
            
            # 标准化 (z-score)
            self.data['fitness_normalized'] = (self.data['DMS_score'] - self.fitness_mean) / self.fitness_std
            
            print(f"\nFitness normalization info:")
            print(f"  Original value range: {self.data['DMS_score'].min():.6f} - {self.data['DMS_score'].max():.6f}")
            print(f"  Original mean: {self.fitness_mean:.6f}")
            print(f"  Original std: {self.fitness_std:.6f}")
            print(f"  Normalized range: {self.data['fitness_normalized'].min():.2f} - {self.data['fitness_normalized'].max():.2f}")
            print(f"  Normalized mean: {self.data['fitness_normalized'].mean():.6f}")
            print(f"  Normalized std: {self.data['fitness_normalized'].std():.6f}")
        
        print(f"\nTotal number of samples: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        
        # 使用标准化后的fitness值
        if self.normalize:
            fitness = row['fitness_normalized']
        else:
            fitness = row['DMS_score']
        
        # 编码序列
        input_ids = self.tokenizer.encode(sequence, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'fitness': torch.tensor(fitness, dtype=torch.float32)
        }
    
    def denormalize(self, normalized_values):
        """
        将标准化后的预测值转换回原始尺度
        
        Args:
            normalized_values: 标准化后的值
            
        Returns:
            原始尺度的值
        """
        if not self.normalize:
            return normalized_values
        
        # 反标准化 (z-score)
        original_values = normalized_values * self.fitness_std + self.fitness_mean
        
        return original_values


def load_rnagym_data(data_dir, dataset_names, tokenizer, batch_size=32, 
                     train_ratio=0.8, max_length=512, num_workers=0, normalize=True):
    """
    Load RNAGym dataset and split into training and validation sets
    
    Args:
        data_dir: Data directory path
        dataset_names: Dataset name(s) (CSV filename without extension) or list of names
        tokenizer: RNA tokenizer
        batch_size: Batch size
        train_ratio: Training set ratio
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        normalize: Whether to normalize fitness values
        
    Returns:
        train_loader, val_loader, full_dataset: Training and validation dataloaders, and full dataset object
    """
    # 支持单个数据集或多个数据集
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # 构建CSV文件路径列表
    csv_files = [os.path.join(data_dir, f"{name}.csv") for name in dataset_names]
    
    # 创建数据集
    full_dataset = RNAGymDataset(csv_files, tokenizer, max_length, normalize=normalize)
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
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
    
    return train_loader, val_loader, full_dataset


def get_available_datasets(data_dir):
    """获取可用的数据集列表"""
    datasets = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            datasets.append(file[:-4])  # 移除.csv扩展名
    return sorted(datasets)
