"""
快速测试脚本：验证模型和数据加载功能
"""
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mamba_rna import create_model
from utils.data_loader import RNATokenizer, load_rnagym_data, get_available_datasets
from utils.metrics import calculate_all_metrics

def test_tokenizer():
    """测试分词器"""
    print("=" * 60)
    print("测试 RNA Tokenizer")
    print("=" * 60)
    
    tokenizer = RNATokenizer()
    
    # 测试编码
    test_seq = "AUGCAUGC"
    encoded = tokenizer.encode(test_seq, max_length=20)
    print(f"原始序列: {test_seq}")
    print(f"编码结果: {encoded[:15]}...")
    
    # 测试解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")
    print("✓ Tokenizer测试通过\n")


def test_model():
    """测试模型"""
    print("=" * 60)
    print("测试 Mamba RNA 模型")
    print("=" * 60)
    
    # 创建模型
    model = create_model()
    
    # 测试前向传播
    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, 8, (batch_size, seq_len))
    
    print(f"输入shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输出shape: {output.shape}")
    print(f"输出示例: {output[:2]}")
    print("✓ 模型测试通过\n")


def test_data_loader():
    """测试数据加载"""
    print("=" * 60)
    print("测试数据加载")
    print("=" * 60)
    
    data_dir = "data/RNAGym"
    
    # 列出可用数据集
    datasets = get_available_datasets(data_dir)
    print(f"找到 {len(datasets)} 个数据集")
    print(f"前3个数据集: {datasets[:3]}")
    
    # 测试加载一个小数据集
    tokenizer = RNATokenizer()
    dataset_name = datasets[0]
    
    print(f"\n加载数据集: {dataset_name}")
    train_loader, val_loader = load_rnagym_data(
        data_dir=data_dir,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=16,
        train_ratio=0.8,
        max_length=256,
        num_workers=0
    )
    
    # 测试一个batch
    batch = next(iter(train_loader))
    print(f"\n批次数据:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  fitness shape: {batch['fitness'].shape}")
    print(f"  fitness 示例: {batch['fitness'][:3]}")
    print("✓ 数据加载测试通过\n")


def test_metrics():
    """测试评估指标"""
    print("=" * 60)
    print("测试评估指标")
    print("=" * 60)
    
    # 生成模拟数据
    import numpy as np
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.3  # 添加噪声
    
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print("✓ 评估指标测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_tokenizer()
        test_model()
        test_data_loader()
        test_metrics()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        print("\n项目已准备就绪，可以开始训练了！")
        print("运行命令: python train.py --epochs 10 --batch_size 16")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
