"""
简化的Mamba SSM实现，用于测试和演示
当无法安装mamba-ssm时，可以使用这个简化版本
"""
import torch
import torch.nn as nn


class SimpleMamba(nn.Module):
    """
    简化的Mamba实现，用于测试
    使用GRU作为替代实现来模拟状态空间模型的行为
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        Args:
            d_model: 模型维度
            d_state: 状态维度（在这个简化版本中对应GRU的隐藏层）
            d_conv: 卷积核大小
            expand: 扩展因子
        """
        super().__init__()
        self.d_model = d_model
        
        # 使用双向GRU模拟状态空间模型
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # GRU处理
        x_gru, _ = self.gru(x)
        
        # 输出投影
        out = self.out_proj(x_gru)
        
        return out


# 创建别名，使其与真实的mamba-ssm兼容
Mamba = SimpleMamba
