"""
快速开始示例
展示如何使用RNA fitness预测框架
"""

import torch
import numpy as np
from rna_fitness import MambaRNA, RNATokenizer, RNADataset
from rna_fitness.data import create_dataloaders


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("=" * 50)
    print("示例1: 基本使用")
    print("=" * 50)
    
    # 初始化tokenizer
    tokenizer = RNATokenizer()
    
    # RNA序列示例
    sequences = [
        "AUGCAUGCAUGC",
        "UUUUAAAACCCC",
        "GGGGUUUUCCCC"
    ]
    
    # 编码序列
    print("\n编码RNA序列:")
    for seq in sequences:
        encoded = tokenizer.encode(seq, max_length=20, return_tensors=True)
        print(f"  {seq} -> {encoded.shape}")
        decoded = tokenizer.decode(encoded[0])
        print(f"    解码: {decoded}")
    
    # 批量编码
    print("\n批量编码:")
    batch = tokenizer.batch_encode(sequences, max_length=20)
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")


def example_2_model_inference():
    """示例2: 模型推理"""
    print("\n" + "=" * 50)
    print("示例2: 模型推理")
    print("=" * 50)
    
    # 初始化tokenizer和模型
    tokenizer = RNATokenizer()
    model = MambaRNA(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        use_mamba=False  # 使用LSTM作为示例
    )
    model.eval()
    
    # 测试序列
    sequences = [
        "AUGCAUGCAUGC",
        "UUUUAAAACCCC",
    ]
    
    # 编码并预测
    print("\n预测fitness值:")
    with torch.no_grad():
        batch = tokenizer.batch_encode(sequences, max_length=50)
        predictions = model(batch['input_ids'], batch['attention_mask'])
        
        for seq, pred in zip(sequences, predictions):
            print(f"  {seq}: {pred.item():.4f}")


def example_3_dataset():
    """示例3: 使用数据集"""
    print("\n" + "=" * 50)
    print("示例3: 使用数据集")
    print("=" * 50)
    
    # 创建示例数据
    sequences = ['AUGC' * 10 for _ in range(50)]
    fitness = np.random.randn(50).tolist()
    
    # 初始化tokenizer
    tokenizer = RNATokenizer()
    
    # 创建数据集
    dataset = RNADataset(
        sequences=sequences,
        fitness=fitness,
        tokenizer=tokenizer,
        max_length=100
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取样本
    sample = dataset[0]
    print(f"样本示例:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  label: {sample['labels'].item():.4f}")


def example_4_dataloader():
    """示例4: 使用数据加载器"""
    print("\n" + "=" * 50)
    print("示例4: 使用数据加载器")
    print("=" * 50)
    
    # 创建示例数据
    train_sequences = ['AUGC' * 10 for _ in range(100)]
    train_fitness = np.random.randn(100).tolist()
    val_sequences = ['AUGC' * 10 for _ in range(20)]
    val_fitness = np.random.randn(20).tolist()
    
    # 初始化tokenizer
    tokenizer = RNATokenizer()
    
    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(
        train_sequences=train_sequences,
        train_fitness=train_fitness,
        val_sequences=val_sequences,
        val_fitness=val_fitness,
        tokenizer=tokenizer,
        batch_size=16,
        max_length=100,
        num_workers=0
    )
    
    print(f"\n训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print(f"\n批次示例:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")


def example_5_training_loop():
    """示例5: 简单的训练循环"""
    print("\n" + "=" * 50)
    print("示例5: 简单的训练循环")
    print("=" * 50)
    
    # 创建示例数据
    train_sequences = ['AUGC' * 10 for _ in range(100)]
    train_fitness = np.random.randn(100).tolist()
    
    # 初始化
    tokenizer = RNATokenizer()
    model = MambaRNA(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    
    # 数据加载器
    train_loader, _, _ = create_dataloaders(
        train_sequences=train_sequences,
        train_fitness=train_fitness,
        tokenizer=tokenizer,
        batch_size=16,
        max_length=100,
        num_workers=0
    )
    
    # 训练设置
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n训练3个step作为示例...")
    model.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只训练3个step
            break
            
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"  Step {i+1}, Loss: {loss.item():.4f}")
    
    print("训练示例完成！")


def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("RNA Fitness预测框架 - 快速开始")
    print("=" * 50)
    
    example_1_basic_usage()
    example_2_model_inference()
    example_3_dataset()
    example_4_dataloader()
    example_5_training_loop()
    
    print("\n" + "=" * 50)
    print("所有示例完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
