"""
RNA Fitness预测模型训练脚本
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from models.mamba_rna import create_model
from utils.data_loader import RNATokenizer, load_rnagym_data, get_available_datasets
from utils.metrics import calculate_all_metrics, print_metrics
from utils.visualization import plot_training_curves, plot_prediction_scatter, save_results_summary


def train_epoch(model, train_loader, criterion, optimizer, device, dataset=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        targets = batch['fitness'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(input_ids)
        
        # Calculate loss (in normalized space)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Denormalize before calculating metrics if normalized
    if dataset is not None and dataset.normalize:
        all_preds_original = dataset.denormalize(all_preds)
        all_targets_original = dataset.denormalize(all_targets)
        metrics = calculate_all_metrics(all_targets_original, all_preds_original)
    else:
        metrics = calculate_all_metrics(all_targets, all_preds)
    
    return avg_loss, metrics


def validate(model, val_loader, criterion, device, dataset=None):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            targets = batch['fitness'].to(device)
            
            # Forward pass
            predictions = model(input_ids)
            
            # Calculate loss (in normalized space)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Denormalize before calculating metrics if normalized
    if dataset is not None and dataset.normalize:
        all_preds_original = dataset.denormalize(all_preds)
        all_targets_original = dataset.denormalize(all_targets)
        metrics = calculate_all_metrics(all_targets_original, all_preds_original)
    else:
        metrics = calculate_all_metrics(all_targets, all_preds)
    
    return avg_loss, metrics, all_targets, all_preds


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = RNATokenizer()
    
    # 加载数据
    print(f"\n加载数据集: {args.datasets}")
    train_loader, val_loader, dataset = load_rnagym_data(
        data_dir=args.data_dir,
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        max_length=args.max_length,
        num_workers=args.num_workers,
        normalize=True  # 启用标准化
    )
    
    # 创建模型
    print("\n创建模型...")
    model_config = {
        'vocab_size': 8,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'd_state': args.d_state,
        'd_conv': args.d_conv,
        'expand': args.expand,
        'dropout': args.dropout
    }
    model = create_model(model_config)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练循环
    print(f"\n开始训练 {args.epochs} 个epoch...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, dataset)
        train_losses.append(train_loss)
        train_metrics_history.append(train_metrics)
        
        print(f"\nTraining Set - Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="Training Set ")
        
        # 验证
        val_loss, val_metrics, val_targets, val_preds = validate(model, val_loader, criterion, device, dataset)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        print(f"\nValidation Set - Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="Validation Set ")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'model_config': model_config
            }, model_path)
            print(f"\n✓ 保存最佳模型到: {model_path}")
        
        # 绘制训练曲线
        if epoch % args.plot_interval == 0 or epoch == args.epochs:
            plot_path = os.path.join(args.output_dir, 'training_curves.png')
            plot_training_curves(
                train_losses, val_losses,
                train_metrics_history, val_metrics_history,
                save_path=plot_path
            )
    
    # 训练结束，加载最佳模型进行最终评估
    print(f"\n{'='*60}")
    print("训练完成！加载最佳模型进行最终评估...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终验证
    _, final_metrics, final_targets, final_preds = validate(model, val_loader, criterion, device, dataset)
    
    print("\nFinal Evaluation Results:")
    print_metrics(final_metrics)
    
    # 反标准化用于可视化
    if dataset.normalize:
        final_targets_original = dataset.denormalize(final_targets)
        final_preds_original = dataset.denormalize(final_preds)
    else:
        final_targets_original = final_targets
        final_preds_original = final_preds
    
    # 保存预测散点图
    scatter_path = os.path.join(args.output_dir, 'prediction_scatter.png')
    plot_prediction_scatter(final_targets_original, final_preds_original, final_metrics, save_path=scatter_path)
    
    # 保存结果摘要
    summary_path = os.path.join(args.output_dir, 'results_summary.txt')
    save_results_summary(final_metrics, save_path=summary_path)
    
    print(f"\n训练完成！所有结果已保存到: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='RNA Fitness Prediction Model Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/RNAGym',
                        help='Data directory path')
    parser.add_argument('--datasets', type=str, nargs='+', default=['Andreasson_2020_ribozyme'],
                        help='Dataset name(s) (can specify multiple)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of Mamba layers')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Convolution kernel size')
    parser.add_argument('--expand', type=int, default=2,
                        help='Expansion factor')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--plot_interval', type=int, default=5,
                        help='Plot interval (epochs)')
    
    args = parser.parse_args()
    
    # Display available datasets
    print("\nAvailable datasets:")
    available_datasets = get_available_datasets(args.data_dir)
    for i, ds in enumerate(available_datasets, 1):
        marker = "✓" if ds in args.datasets else " "
        print(f"  [{marker}] {i}. {ds}")
    
    print(f"\nCurrently selected dataset(s): {', '.join(args.datasets)}")
    
    train(args)


if __name__ == '__main__':
    main()
