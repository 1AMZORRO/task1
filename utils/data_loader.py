"""
RNA序列数据加载和预处理模块
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


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
    """RNAGym数据集 - 支持fitness标准化"""
    
    def __init__(self, csv_file, tokenizer, max_length=512, fitness_mean=None, fitness_std=None):
        """
        Args:
            csv_file: CSV文件路径
            tokenizer: RNA分词器
            max_length: 最大序列长度
            fitness_mean: Fitness均值（用于标准化），如果为None则不标准化
            fitness_std: Fitness标准差（用于标准化），如果为None则不标准化
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fitness_mean = fitness_mean
        self.fitness_std = fitness_std
        
        # 检查数据
        print(f"加载数据集: {csv_file}")
        print(f"  样本数量: {len(self.data)}")
        print(f"  序列长度范围: {self.data['sequence'].str.len().min()} - {self.data['sequence'].str.len().max()}")
        
        # 打印fitness统计信息
        raw_fitness = self.data['DMS_score'].values
        print(f"  DMS_score统计:")
        print(f"    - 均值: {np.mean(raw_fitness):.6f}")
        print(f"    - 标准差: {np.std(raw_fitness):.6f}")
        print(f"    - 最小值: {np.min(raw_fitness):.6f}")
        print(f"    - 最大值: {np.max(raw_fitness):.6f}")
        
        if self.fitness_mean is not None and self.fitness_std is not None:
            print(f"  Fitness标准化已启用:")
            print(f"    - 标准化均值: {self.fitness_mean:.6f}")
            print(f"    - 标准化标准差: {self.fitness_std:.6f}")
        else:
            print(f"  Fitness标准化: 未启用")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        fitness = row['DMS_score']
        
        # 标准化fitness（如果提供了均值和标准差）
        if self.fitness_mean is not None and self.fitness_std is not None:
            if self.fitness_std > 0:
                fitness = (fitness - self.fitness_mean) / self.fitness_std
            else:
                # 如果标准差为0，只减去均值
                fitness = fitness - self.fitness_mean
        
        # 编码序列
        input_ids = self.tokenizer.encode(sequence, self.max_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'fitness': torch.tensor(fitness, dtype=torch.float32)
        }


def load_rnagym_data(data_dir, dataset_name, tokenizer, batch_size=32, 
                     train_ratio=0.8, max_length=512, num_workers=0, normalize_fitness=True):
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
        normalize_fitness: 是否标准化fitness值（推荐True）
        
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
        fitness_stats: fitness统计信息字典 {'mean': float, 'std': float}
    """
    csv_file = os.path.join(data_dir, f"{dataset_name}.csv")
    
    # 首先读取完整数据集以计算统计信息
    df = pd.read_csv(csv_file)
    total_size = len(df)
    
    # 创建分层列：使用DMS_score的分位数分箱
    try:
        # 尝试使用qcut进行等频分箱（推荐方式）
        df['stratify_col'] = pd.qcut(df['DMS_score'], q=10, labels=False, duplicates='drop')
    except (ValueError, TypeError) as e:
        # 如果qcut失败（例如重复值太多），回退到cut进行等宽分箱
        print(f"  注意: pd.qcut失败 ({e})，使用pd.cut作为备选方案")
        df['stratify_col'] = pd.cut(df['DMS_score'], bins=10, labels=False)
    
    # 处理可能的NaN值（用-1填充）
    df['stratify_col'] = df['stratify_col'].fillna(-1)
    
    # 使用分层分割来划分训练集和验证集
    indices = np.arange(total_size)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        stratify=df['stratify_col'],
        random_state=42
    )
    
    # 删除临时分层列以释放内存
    df.drop('stratify_col', axis=1, inplace=True)
    
    # 计算训练集的fitness统计信息（用于标准化）
    train_fitness = df.iloc[train_indices]['DMS_score'].values
    fitness_mean = float(np.mean(train_fitness))
    fitness_std = float(np.std(train_fitness))
    
    print("\n" + "="*80)
    print("Fitness标准化统计信息 (基于训练集计算)")
    print("="*80)
    print(f"  训练集均值: {fitness_mean:.6f}")
    print(f"  训练集标准差: {fitness_std:.6f}")
    print(f"  标准化: {'已启用' if normalize_fitness else '未启用'}")
    print("="*80 + "\n")
    
    # 创建数据集（使用训练集的统计信息进行标准化）
    if normalize_fitness:
        full_dataset = RNAGymDataset(csv_file, tokenizer, max_length, 
                                    fitness_mean=fitness_mean, 
                                    fitness_std=fitness_std)
    else:
        full_dataset = RNAGymDataset(csv_file, tokenizer, max_length,
                                    fitness_mean=None,
                                    fitness_std=None)
    
    # 使用相同的划分索引
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
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
    
    # 返回统计信息以便后续反标准化
    fitness_stats = {
        'mean': fitness_mean,
        'std': fitness_std,
        'normalize': normalize_fitness
    }
    
    return train_loader, val_loader, fitness_stats


def get_available_datasets(data_dir):
    """获取可用的数据集列表"""
    datasets = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            datasets.append(file[:-4])  # 移除.csv扩展名
    return sorted(datasets)
