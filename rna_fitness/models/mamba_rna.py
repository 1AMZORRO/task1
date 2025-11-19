"""
基于Mamba SSM的RNA fitness预测模型
"""

import torch
import torch.nn as nn
from typing import Optional


class MambaRNA(nn.Module):
    """
    使用Mamba状态空间模型进行RNA fitness预测
    
    Mamba模型是一种高效的序列建模架构，特别适合处理长序列数据
    """
    
    def __init__(
        self,
        vocab_size: int = 8,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
    ):
        """
        初始化Mamba RNA模型
        
        Args:
            vocab_size: 词汇表大小（RNA为8：4个碱基+4个特殊token）
            d_model: 模型维度
            n_layers: Mamba层数
            d_state: 状态空间维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dropout: dropout比率
            use_mamba: 是否使用Mamba（False时使用简单的LSTM作为备选）
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_mamba = use_mamba
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码（可选）
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Mamba layers或LSTM作为备选
        if use_mamba:
            try:
                from mamba_ssm import Mamba
                self.encoder = nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(n_layers)
                ])
                self.use_actual_mamba = True
            except ImportError:
                print("警告: mamba-ssm未安装，使用LSTM作为备选")
                self.use_actual_mamba = False
                self.encoder = nn.LSTM(
                    d_model,
                    d_model,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0,
                    bidirectional=True
                )
                # 双向LSTM需要投影回d_model
                self.projection = nn.Linear(d_model * 2, d_model)
        else:
            self.use_actual_mamba = False
            self.encoder = nn.LSTM(
                d_model,
                d_model,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=True
            )
            self.projection = nn.Linear(d_model * 2, d_model)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层：回归头，预测fitness值
        self.fitness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID，shape: (batch_size, seq_len)
            attention_mask: 注意力mask，shape: (batch_size, seq_len)
            
        Returns:
            fitness预测值，shape: (batch_size, 1)
        """
        # Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 通过编码器
        if self.use_actual_mamba:
            # Mamba layers
            for layer in self.encoder:
                x = layer(x) + x  # 残差连接
                x = self.dropout(x)
        else:
            # LSTM
            original_seq_len = x.size(1)
            if attention_mask is not None:
                # 计算实际序列长度
                lengths = attention_mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
            
            x, _ = self.encoder(x)
            
            if attention_mask is not None:
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x, batch_first=True, total_length=original_seq_len
                )
            
            x = self.projection(x)
        
        # Layer norm
        x = self.norm(x)
        
        # 池化：使用mean pooling
        if attention_mask is not None:
            # Mask pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            x = x.mean(dim=1)  # (batch_size, d_model)
        
        # 预测fitness
        fitness = self.fitness_head(x)  # (batch_size, 1)
        
        return fitness.squeeze(-1)  # (batch_size,)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
