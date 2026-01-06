"""
基于Mamba SSM的RNA Fitness预测模型
"""
import torch
import torch.nn as nn

# 尝试导入mamba-ssm，如果失败则使用简化版本
try:
    from mamba_ssm import Mamba
    print("使用官方mamba-ssm实现")
except ImportError:
    from .simple_mamba import Mamba
    print("警告: mamba-ssm未安装，使用简化版本（基于GRU）")


class MambaBlock(nn.Module):
    """Mamba SSM Block"""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """
        Args:
            d_model: 模型维度
            d_state: SSM状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
        """
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return x + self.mamba(self.norm(x))


class MambaRNAModel(nn.Module):
    """基于Mamba的RNA Fitness预测模型"""
    
    def __init__(
        self,
        vocab_size=8,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_layers: Mamba层数
            d_state: SSM状态维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dropout: Dropout率
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Token嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（可选，Mamba本身对位置敏感）
        self.pos_dropout = nn.Dropout(dropout)
        
        # Mamba层堆叠
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 回归头 - 预测fitness score
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len) - token IDs
            
        Returns:
            predictions: (batch,) - 预测的fitness分数
        """
        # 嵌入
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_dropout(x)
        
        # 通过Mamba层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 池化：使用CLS token（第一个token）或平均池化
        # 这里使用CLS token
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # 回归预测
        predictions = self.regression_head(cls_output)  # (batch, 1)
        
        return predictions.squeeze(-1)  # (batch,)
    
    def get_num_params(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config=None):
    """
    创建模型的工厂函数
    
    Args:
        config: 配置字典，如果为None则使用默认配置
        
    Returns:
        model: MambaRNAModel实例
    """
    if config is None:
        config = {
            'vocab_size': 8,
            'd_model': 256,
            'n_layers': 4,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.1
        }
    
    model = MambaRNAModel(**config)
    print(f"模型参数量: {model.get_num_params():,}")
    
    return model
