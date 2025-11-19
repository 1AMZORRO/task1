"""
训练脚本
使用Mamba模型训练RNA fitness预测
"""

import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rna_fitness import MambaRNA, RNADataset, RNATokenizer
from rna_fitness.data import create_dataloaders
from rna_fitness.utils.metrics import compute_metrics, print_metrics


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions_list = []
    labels_list = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [训练]")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        predictions_list.extend(outputs.detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(predictions_list, labels_list)
    metrics['loss'] = avg_loss
    
    return metrics


def evaluate(model, data_loader, criterion, device, desc="验证"):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 记录
            total_loss += loss.item()
            predictions_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(predictions_list, labels_list)
    metrics['loss'] = avg_loss
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练RNA fitness预测模型')
    parser.add_argument('--config', type=str, default='rna_fitness/configs/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train_data', type=str, default=None,
                        help='训练数据路径')
    parser.add_argument('--val_data', type=str, default=None,
                        help='验证数据路径')
    parser.add_argument('--test_data', type=str, default=None,
                        help='测试数据路径')
    args = parser.parse_args()
    
    # 加载配置
    print("加载配置文件...")
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['result_dir'], exist_ok=True)
    
    # 初始化tokenizer
    print("初始化tokenizer...")
    tokenizer = RNATokenizer()
    
    # 加载数据
    print("加载数据...")
    train_data_path = args.train_data or config['data'].get('train_path')
    val_data_path = args.val_data or config['data'].get('val_path')
    test_data_path = args.test_data or config['data'].get('test_path')
    
    if train_data_path is None:
        print("警告: 未提供训练数据，使用示例数据进行演示")
        # 创建示例数据
        train_sequences = ['AUGC' * 10 for _ in range(100)]
        train_fitness = np.random.randn(100).tolist()
        val_sequences = ['AUGC' * 10 for _ in range(20)]
        val_fitness = np.random.randn(20).tolist()
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_sequences=train_sequences,
            train_fitness=train_fitness,
            val_sequences=val_sequences,
            val_fitness=val_fitness,
            tokenizer=tokenizer,
            batch_size=config['data']['batch_size'],
            max_length=config['data']['max_length'],
            num_workers=0  # 示例数据使用0个worker
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
            tokenizer=tokenizer,
            batch_size=config['data']['batch_size'],
            max_length=config['data']['max_length'],
            num_workers=config['data']['num_workers']
        )
    
    # 初始化模型
    print("初始化模型...")
    model = MambaRNA(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        d_state=config['model']['d_state'],
        d_conv=config['model']['d_conv'],
        expand=config['model']['expand'],
        dropout=config['model']['dropout'],
        use_mamba=config['model']['use_mamba']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # TensorBoard
    writer = None
    if config['output']['use_tensorboard']:
        writer = SummaryWriter(config['output']['log_dir'])
    
    # 训练循环
    print("\n开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"\nEpoch {epoch} - 训练集:")
        print_metrics(train_metrics, prefix="  ")
        
        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f'train/{key}', value, epoch)
        
        # 验证
        if val_loader and epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, desc="验证")
            print(f"\nEpoch {epoch} - 验证集:")
            print_metrics(val_metrics, prefix="  ")
            
            if writer:
                for key, value in val_metrics.items():
                    writer.add_scalar(f'val/{key}', value, epoch)
            
            # 早停和模型保存
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # 保存最佳模型
                checkpoint_path = os.path.join(
                    config['output']['checkpoint_dir'],
                    'best_model.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config
                }, checkpoint_path)
                print(f"  保存最佳模型到: {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"\n早停触发！验证损失已经{patience_counter}个epoch没有改善。")
                break
        
        # 定期保存检查点
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['output']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
    
    # 测试
    if test_loader:
        print("\n在测试集上评估...")
        # 加载最佳模型
        checkpoint_path = os.path.join(config['output']['checkpoint_dir'], 'best_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型 (epoch {checkpoint['epoch']})")
        
        test_metrics = evaluate(model, test_loader, criterion, device, desc="测试")
        print("\n测试集结果:")
        print_metrics(test_metrics)
        
        if writer:
            for key, value in test_metrics.items():
                writer.add_scalar(f'test/{key}', value, 0)
    
    if writer:
        writer.close()
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
