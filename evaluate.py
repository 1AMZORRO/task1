"""
评估脚本
评估训练好的RNA fitness预测模型
"""

import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from rna_fitness import MambaRNA, RNATokenizer
from rna_fitness.data import create_dataloaders
from rna_fitness.utils.metrics import compute_metrics, print_metrics


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, data_loader, device, save_predictions=None):
    """评估模型并可选择保存预测结果"""
    model.eval()
    predictions_list = []
    labels_list = []
    sequences_list = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="评估中")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            
            # 记录结果
            predictions_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    # 计算指标
    metrics = compute_metrics(predictions_list, labels_list)
    
    # 保存预测结果
    if save_predictions:
        results_df = pd.DataFrame({
            'true_fitness': labels_list,
            'predicted_fitness': predictions_list,
            'error': np.array(predictions_list) - np.array(labels_list)
        })
        results_df.to_csv(save_predictions, index=False)
        print(f"\n预测结果已保存到: {save_predictions}")
    
    return metrics, predictions_list, labels_list


def main():
    parser = argparse.ArgumentParser(description='评估RNA fitness预测模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（可选，会从checkpoint中读取）')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据路径')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='预测结果输出路径')
    args = parser.parse_args()
    
    # 加载检查点
    print(f"加载模型检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config')
        if config is None:
            raise ValueError("检查点中没有配置信息，请通过--config参数提供配置文件")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化tokenizer
    print("初始化tokenizer...")
    tokenizer = RNATokenizer()
    
    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    _, _, test_loader = create_dataloaders(
        test_data_path=args.test_data,
        tokenizer=tokenizer,
        batch_size=config['data']['batch_size'],
        max_length=config['data']['max_length'],
        num_workers=config['data']['num_workers']
    )
    
    if test_loader is None:
        raise ValueError("无法加载测试数据")
    
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
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 评估
    print("\n开始评估...")
    metrics, predictions, labels = evaluate_model(
        model, test_loader, device, save_predictions=args.output
    )
    
    # 打印结果
    print("\n=== 评估结果 ===")
    print_metrics(metrics)
    
    # 额外的统计信息
    print("\n=== 预测统计 ===")
    print(f"预测值范围: [{min(predictions):.4f}, {max(predictions):.4f}]")
    print(f"真实值范围: [{min(labels):.4f}, {max(labels):.4f}]")
    print(f"平均预测值: {np.mean(predictions):.4f}")
    print(f"平均真实值: {np.mean(labels):.4f}")
    print(f"预测值标准差: {np.std(predictions):.4f}")
    print(f"真实值标准差: {np.std(labels):.4f}")


if __name__ == '__main__':
    main()
