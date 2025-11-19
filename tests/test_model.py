"""
测试Mamba RNA模型
"""

import torch
import pytest
from rna_fitness.models.mamba_rna import MambaRNA


def test_model_init():
    """测试模型初始化"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False  # 使用LSTM以避免依赖问题
    )
    
    assert model.vocab_size == 8
    assert model.d_model == 64
    assert model.n_layers == 2


def test_model_forward():
    """测试模型前向传播"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    
    # 创建输入
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 8, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    outputs = model(input_ids, attention_mask)
    
    # 检查输出形状
    assert outputs.shape == (batch_size,)


def test_model_output_range():
    """测试模型输出范围"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    model.eval()
    
    # 创建输入
    input_ids = torch.randint(0, 8, (4, 20))
    attention_mask = torch.ones(4, 20)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # 输出应该是实数（fitness值可以是任何实数）
    assert torch.isfinite(outputs).all()


def test_model_different_lengths():
    """测试不同长度的序列"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    
    # 不同长度的序列（通过attention mask控制）
    input_ids = torch.randint(0, 8, (2, 30))
    attention_mask = torch.zeros(2, 30)
    attention_mask[0, :10] = 1  # 第一个序列长度10
    attention_mask[1, :20] = 1  # 第二个序列长度20
    
    outputs = model(input_ids, attention_mask)
    
    assert outputs.shape == (2,)
    assert torch.isfinite(outputs).all()


def test_model_parameters():
    """测试模型参数数量"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    
    # 应该有一定数量的参数
    assert num_params > 0
    print(f"模型参数量: {num_params:,}")


def test_model_gradient_flow():
    """测试梯度流"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        use_mamba=False
    )
    
    # 创建输入
    input_ids = torch.randint(0, 8, (4, 20))
    attention_mask = torch.ones(4, 20)
    labels = torch.randn(4)
    
    # 前向传播
    outputs = model(input_ids, attention_mask)
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(outputs, labels)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 没有梯度"
            assert torch.isfinite(param.grad).all(), f"参数 {name} 的梯度包含NaN或Inf"


def test_model_eval_mode():
    """测试评估模式"""
    model = MambaRNA(
        vocab_size=8,
        d_model=64,
        n_layers=2,
        dropout=0.5,
        use_mamba=False
    )
    
    input_ids = torch.randint(0, 8, (2, 20))
    attention_mask = torch.ones(2, 20)
    
    # 训练模式
    model.train()
    output1 = model(input_ids, attention_mask)
    output2 = model(input_ids, attention_mask)
    
    # 由于dropout，两次输出可能不同
    # 但评估模式下应该相同
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        output3 = model(input_ids, attention_mask)
        output4 = model(input_ids, attention_mask)
    
    # 评估模式下输出应该相同
    assert torch.allclose(output3, output4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
