# RNA Fitness预测框架架构说明

## 概述

本框架是一个基于深度学习的RNA fitness预测系统，核心采用Mamba状态空间模型（SSM）。框架设计遵循模块化原则，易于扩展和维护。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户接口层                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ train.py │  │evaluate.py│  │examples/ │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    核心模块层                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │           rna_fitness 包                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │  │
│  │  │  models  │  │   data   │  │  utils   │       │  │
│  │  └──────────┘  └──────────┘  └──────────┘       │  │
│  │       │              │              │            │  │
│  │  ┌────▼────┐    ┌───▼────┐    ┌───▼────┐       │  │
│  │  │ Mamba   │    │Dataset │    │Tokenizer│      │  │
│  │  │  RNA    │    │        │    │         │      │  │
│  │  └─────────┘    └────────┘    └─────────┘      │  │
│  │                                                  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │         configs (YAML配置)                │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    依赖库层                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ PyTorch  │  │Mamba-SSM │  │  NumPy   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 模型层 (models/)

#### MambaRNA 模型

**文件**: `rna_fitness/models/mamba_rna.py`

**核心功能**:
- RNA序列到fitness值的端到端预测
- 支持Mamba SSM和LSTM两种架构
- 可变长度序列处理

**架构细节**:

```
输入: RNA序列 (token IDs)
  │
  ▼
┌────────────────────┐
│  Embedding Layer   │  将token ID转换为向量
└────────────────────┘
  │
  ▼
┌────────────────────┐
│ Positional Encoding│  添加位置信息
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  Encoder Layers    │  Mamba/LSTM层
│  (n_layers)        │  特征提取
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  Layer Norm        │  归一化
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  Mean Pooling      │  序列聚合
└────────────────────┘
  │
  ▼
┌────────────────────┐
│  Fitness Head      │  MLP回归头
└────────────────────┘
  │
  ▼
输出: Fitness值 (标量)
```

**关键参数**:
- `vocab_size`: 词汇表大小（8：4个碱基+4个特殊token）
- `d_model`: 隐藏层维度
- `n_layers`: 编码器层数
- `d_state`: Mamba状态空间维度
- `dropout`: Dropout比率

**Mamba vs LSTM**:

| 特性 | Mamba SSM | LSTM |
|------|-----------|------|
| 速度 | 更快 | 较慢 |
| 内存 | 更小 | 较大 |
| 长序列 | 优秀 | 较好 |
| 依赖 | 需要编译 | 标准PyTorch |
| 使用场景 | 生产环境 | 快速原型 |

### 2. 数据层 (data/)

#### RNADataset

**文件**: `rna_fitness/data/dataset.py`

**功能**:
- 加载和处理RNA序列数据
- 支持CSV/TSV格式
- 自动序列编码和填充
- 集成tokenizer

**数据流**:

```
原始数据 (CSV/TSV)
  │
  ▼
┌─────────────────┐
│  数据加载        │  读取文件
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  序列编码        │  Tokenizer
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  填充/截断       │  统一长度
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  Batch整理      │  DataLoader
└─────────────────┘
  │
  ▼
模型输入
```

#### DataLoader创建

**函数**: `create_dataloaders()`

**功能**:
- 创建训练/验证/测试数据加载器
- 自动配置batch size和workers
- 支持shuffle和pin_memory

### 3. 工具层 (utils/)

#### RNATokenizer

**文件**: `rna_fitness/utils/tokenizer.py`

**核心功能**:
- RNA序列到token ID的转换
- DNA自动转换为RNA（T→U）
- 批量编码/解码
- Attention mask生成

**词汇表**:

```python
{
    '<PAD>': 0,   # 填充
    '<UNK>': 1,   # 未知
    '<CLS>': 2,   # 序列开始
    '<SEP>': 3,   # 序列结束
    'A': 4,       # 腺嘌呤
    'U': 5,       # 尿嘧啶
    'G': 6,       # 鸟嘌呤
    'C': 7,       # 胞嘧啶
}
```

**编码流程**:

```
RNA序列: "AUGC"
  │
  ▼ 添加特殊token
[<CLS>, A, U, G, C, <SEP>]
  │
  ▼ 转换为ID
[2, 4, 5, 6, 7, 3]
  │
  ▼ 填充到max_length
[2, 4, 5, 6, 7, 3, 0, 0, ..., 0]
  │
  ▼ 生成attention mask
[1, 1, 1, 1, 1, 1, 0, 0, ..., 0]
```

#### Metrics

**文件**: `rna_fitness/utils/metrics.py`

**RNAGym标准评估指标**（主要）:

1. **Spearman相关系数（绝对值）**
   ```python
   spearman = abs(spearmanr(predictions, labels).correlation)
   ```
   - 秩相关，衡量单调关系
   - 范围[0, 1]（取绝对值）
   - 越接近1越好

2. **AUC (Area Under ROC Curve)**
   ```python
   # 使用中位数作为阈值二分类
   binary_labels = (labels > median(labels)).astype(int)
   auc = max(roc_auc_score(binary_labels, predictions), 1 - auc)
   ```
   - ROC曲线下面积
   - 范围[0.5, 1.0]（取max(auc, 1-auc)确保>=0.5）
   - 越接近1越好

3. **MCC (Matthews Correlation Coefficient，绝对值)**
   ```python
   binary_labels = (labels > median(labels)).astype(int)
   binary_preds = (predictions > median(predictions)).astype(int)
   mcc = abs(matthews_corrcoef(binary_labels, binary_preds))
   ```
   - 马修斯相关系数，评估二分类质量
   - 范围[0, 1]（取绝对值）
   - 越接近1越好

**额外提供的指标**:

4. **MSE (Mean Squared Error)**
   ```
   MSE = (1/n) Σ(y_pred - y_true)²
   ```
   - 衡量预测值与真实值的平均平方误差
   - 越小越好

5. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = √MSE
   ```
   - MSE的平方根，与原始数据同量级

6. **R² (R-squared)**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   - 决定系数，范围[0, 1]
   - 越接近1越好

7. **Pearson相关系数**
   - 线性相关
   - 范围[-1, 1]

> **注意**: RNAGym使用Spearman (abs), AUC, MCC作为主要评估指标，这与论文中的标准一致。

### 4. 配置层 (configs/)

**文件**: `rna_fitness/configs/default_config.yaml`

**配置结构**:

```yaml
model:          # 模型架构参数
  - vocab_size
  - d_model
  - n_layers
  - ...

data:           # 数据处理参数
  - max_length
  - batch_size
  - num_workers
  - ...

training:       # 训练超参数
  - num_epochs
  - learning_rate
  - weight_decay
  - ...

optimizer:      # 优化器配置
  - type
  - betas
  - eps

scheduler:      # 学习率调度
  - type
  - warmup_steps
  - ...

output:         # 输出配置
  - log_dir
  - checkpoint_dir
  - ...
```

## 训练流程

### 完整训练Pipeline

```
1. 初始化
   ├─ 加载配置
   ├─ 设置随机种子
   ├─ 选择设备(CPU/GPU)
   └─ 创建输出目录

2. 数据准备
   ├─ 初始化Tokenizer
   ├─ 加载数据集
   └─ 创建DataLoader

3. 模型初始化
   ├─ 创建MambaRNA模型
   ├─ 定义损失函数(MSE)
   ├─ 配置优化器(AdamW)
   └─ 设置学习率调度器

4. 训练循环
   For each epoch:
   ├─ 训练阶段
   │  For each batch:
   │  ├─ 前向传播
   │  ├─ 计算损失
   │  ├─ 反向传播
   │  ├─ 梯度裁剪
   │  └─ 更新参数
   │
   ├─ 验证阶段
   │  ├─ 计算验证损失
   │  ├─ 计算评估指标
   │  └─ 记录到TensorBoard
   │
   └─ 检查点保存
      ├─ 保存最佳模型
      ├─ 定期保存检查点
      └─ 早停检查

5. 测试评估
   ├─ 加载最佳模型
   ├─ 在测试集上评估
   └─ 保存预测结果
```

### 训练优化技术

1. **梯度裁剪**: 防止梯度爆炸
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **早停**: 防止过拟合
   - 监控验证损失
   - patience轮没有改善则停止

3. **学习率调度**: 优化收敛
   - Warmup阶段
   - Cosine/Linear衰减

4. **权重衰减**: L2正则化
   - 减少模型复杂度
   - 提高泛化能力

## 评估流程

```
1. 加载模型
   ├─ 读取检查点
   └─ 恢复模型状态

2. 数据准备
   ├─ 加载测试数据
   └─ 创建DataLoader

3. 推理
   For each batch:
   ├─ 前向传播
   └─ 收集预测结果

4. 计算指标
   ├─ MSE/RMSE
   ├─ R²
   ├─ Spearman相关
   └─ Pearson相关

5. 保存结果
   ├─ 输出指标
   └─ 保存预测CSV
```

## 扩展性设计

### 添加新模型

```python
# 在 rna_fitness/models/ 中
class NewModel(nn.Module):
    def __init__(self, vocab_size, ...):
        super().__init__()
        # 实现模型架构
        
    def forward(self, input_ids, attention_mask):
        # 实现前向传播
        return fitness_predictions
```

### 添加新的评估指标

```python
# 在 rna_fitness/utils/metrics.py 中
def new_metric(predictions, labels):
    # 计算新指标
    return metric_value

# 更新 compute_metrics 函数
def compute_metrics(predictions, labels):
    metrics = {
        # ... 现有指标
        'new_metric': new_metric(predictions, labels)
    }
    return metrics
```

### 支持新的数据格式

```python
# 在 RNADataset._load_from_file 中添加
elif data_path.endswith('.json'):
    # 处理JSON格式
    pass
```

## 性能优化

### 内存优化

1. **梯度累积**: 小batch size训练大模型
2. **混合精度训练**: 使用FP16减少内存
3. **检查点**: 重计算减少内存占用

### 速度优化

1. **DataLoader**: 多进程数据加载
2. **Pin Memory**: 加速GPU数据传输
3. **编译优化**: 使用torch.compile (PyTorch 2.0+)

### 分布式训练

未来可以支持：
- DataParallel: 单机多GPU
- DistributedDataParallel: 多机多GPU
- FSDP: 大模型训练

## 依赖关系

```
核心依赖:
├── torch (>= 2.0.0)        # 深度学习框架
├── numpy (>= 1.24.0)       # 数值计算
├── pandas (>= 2.0.0)       # 数据处理
└── scikit-learn (>= 1.3.0) # 评估指标

可选依赖:
├── mamba-ssm (>= 1.0.0)    # Mamba模型
├── causal-conv1d (>= 1.1.0) # Mamba依赖
├── tensorboard (>= 2.13.0)  # 可视化
└── wandb (>= 0.15.0)        # 实验跟踪

开发依赖:
├── pytest (>= 7.4.0)       # 测试
├── black (>= 23.0.0)       # 代码格式化
└── flake8 (>= 6.0.0)       # 代码检查
```

## 测试架构

```
tests/
├── test_tokenizer.py    # Tokenizer单元测试
│   ├── 编码解码测试
│   ├── 批量处理测试
│   └── 边界情况测试
│
├── test_model.py        # 模型单元测试
│   ├── 初始化测试
│   ├── 前向传播测试
│   ├── 梯度流测试
│   └── 不同长度序列测试
│
└── test_dataset.py      # 数据集单元测试
    ├── 数据加载测试
    ├── 迭代测试
    └── Attention mask测试
```

## 参考文献

1. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
2. **RNAGym**: RNA fitness prediction benchmark
3. **Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)
4. **LSTM**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
