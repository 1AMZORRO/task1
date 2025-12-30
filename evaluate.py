"""
评估训练好的模型
"""
import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mamba_rna import create_model
from utils.data_loader import RNATokenizer, load_rnagym_data
from utils.metrics import calculate_all_metrics, print_metrics
from utils.visualization import plot_prediction_scatter, save_results_summary
import numpy as np
from tqdm import tqdm


def evaluate_model(model_path, data_dir, dataset_name, output_dir):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model_config = checkpoint['model_config']
    
    # 创建模型
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n加载模型: {model_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.6f}")
    
    # 加载数据
    tokenizer = RNATokenizer()
    train_loader, val_loader = load_rnagym_data(
        data_dir=data_dir,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=32,
        train_ratio=0.8,
        max_length=512,
        num_workers=0
    )
    
    # 评估
    print("\n开始评估...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='评估中')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['fitness'].to(device)
            
            predictions = model(input_ids)
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 计算指标
    metrics = calculate_all_metrics(all_targets, all_preds)
    
    print("\n最终评估结果:")
    print_metrics(metrics)
    
    # 保存可视化
    os.makedirs(output_dir, exist_ok=True)
    scatter_path = os.path.join(output_dir, 'prediction_scatter.png')
    plot_prediction_scatter(all_targets, all_preds, metrics, save_path=scatter_path)
    
    # 保存结果摘要
    summary_path = os.path.join(output_dir, 'results_summary.txt')
    save_results_summary(metrics, save_path=summary_path)
    
    print(f"\n所有结果已保存到: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--model_path', type=str, default='outputs_test/best_model.pt',
                        help='模型文件路径')
    parser.add_argument('--data_dir', type=str, default='data/RNAGym',
                        help='数据目录')
    parser.add_argument('--dataset', type=str, default='Andreasson_2020_ribozyme',
                        help='数据集名称')
    parser.add_argument('--output_dir', type=str, default='outputs_test',
                        help='输出目录')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.dataset, args.output_dir)
