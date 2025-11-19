# RNA Fitness预测框架使用指南

## 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型评估](#模型评估)
5. [自定义配置](#自定义配置)
6. [API使用](#api使用)
7. [常见问题](#常见问题)

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/1AMZORRO/task1.git
cd task1

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 运行示例

```bash
# 运行快速开始示例
python examples/quick_start.py

# 运行测试
pytest tests/ -v
```

## 数据准备

### 数据格式

框架支持CSV和TSV格式的数据文件，必须包含以下两列：

- `sequence`: RNA序列（A, U, G, C）
- `fitness`: fitness值（浮点数）

示例数据格式（`data.csv`）：

```csv
sequence,fitness
AUGCAUGCAUGCAUGC,0.523
UUUUAAAACCCCGGGG,0.876
GGGGCCCCAAAAUUUU,0.234
AUGCUUUUGGGGCCCC,0.645
```

### DNA到RNA转换

框架自动支持DNA到RNA的转换（T -> U），所以你也可以使用DNA序列：

```csv
sequence,fitness
ATGCATGCATGCATGC,0.523
```

将被自动转换为：

```
AUGCAUGCAUGCAUGC
```

### 数据划分

建议将数据划分为训练集、验证集和测试集：

```bash
# 示例目录结构
data/
├── train.csv      # 训练数据（~70%）
├── val.csv        # 验证数据（~15%）
└── test.csv       # 测试数据（~15%）
```

## 模型训练

### 基本训练

使用默认配置训练：

```bash
python train.py \
    --config rna_fitness/configs/default_config.yaml \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --test_data data/test.csv
```

### 使用示例数据训练

如果没有真实数据，可以使用生成的示例数据：

```bash
python train.py --config rna_fitness/configs/default_config.yaml
```

这将使用随机生成的示例数据进行训练演示。

### 训练输出

训练过程中会生成以下输出：

```
logs/                  # TensorBoard日志
checkpoints/           # 模型检查点
  ├── best_model.pt   # 最佳模型
  └── checkpoint_epoch_*.pt  # 定期保存的检查点
results/               # 评估结果
```

### 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir logs/
```

然后在浏览器中打开 `http://localhost:6006`

## 模型评估

### 评估已训练模型

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/test.csv \
    --output predictions.csv
```

### 评估输出

评估会输出以下信息：

1. **评估指标**：
   - MSE (均方误差)
   - RMSE (均方根误差)
   - R² (决定系数)
   - Spearman相关系数
   - Pearson相关系数

2. **预测结果文件** (`predictions.csv`)：
   ```csv
   true_fitness,predicted_fitness,error
   0.523,0.512,-0.011
   0.876,0.889,0.013
   ```

### 示例输出

```
=== 评估结果 ===
  MSE:              0.012345
  RMSE:             0.111111
  R²:               0.892341
  Spearman相关系数:  0.945234 (p=1.23e-10)
  Pearson相关系数:   0.934567 (p=2.45e-09)

=== 预测统计 ===
预测值范围: [0.234, 0.889]
真实值范围: [0.234, 0.876]
平均预测值: 0.567
平均真实值: 0.570
```

## 自定义配置

### 配置文件结构

配置文件位于 `rna_fitness/configs/default_config.yaml`：

```yaml
# 模型配置
model:
  vocab_size: 8      # 词汇表大小（固定为8）
  d_model: 256       # 模型维度（建议：128-512）
  n_layers: 4        # 层数（建议：2-8）
  d_state: 16        # 状态空间维度（Mamba特有）
  d_conv: 4          # 卷积核大小
  expand: 2          # 扩展因子
  dropout: 0.1       # Dropout率（0.0-0.5）
  use_mamba: true    # 是否使用Mamba（false使用LSTM）

# 数据配置
data:
  max_length: 512    # 最大序列长度
  batch_size: 32     # 批次大小
  num_workers: 4     # 数据加载线程数

# 训练配置
training:
  num_epochs: 100              # 训练轮数
  learning_rate: 0.001         # 学习率
  weight_decay: 0.01           # 权重衰减
  gradient_clip: 1.0           # 梯度裁剪
  early_stopping_patience: 10  # 早停耐心值
```

### 创建自定义配置

复制并修改默认配置：

```bash
cp rna_fitness/configs/default_config.yaml my_config.yaml
# 编辑 my_config.yaml
python train.py --config my_config.yaml --train_data data/train.csv
```

### 推荐配置

**小规模数据（< 1000样本）**：
```yaml
model:
  d_model: 128
  n_layers: 2
  dropout: 0.2
training:
  batch_size: 16
  learning_rate: 0.0005
```

**中等规模数据（1000-10000样本）**：
```yaml
model:
  d_model: 256
  n_layers: 4
  dropout: 0.1
training:
  batch_size: 32
  learning_rate: 0.001
```

**大规模数据（> 10000样本）**：
```yaml
model:
  d_model: 512
  n_layers: 6
  dropout: 0.1
training:
  batch_size: 64
  learning_rate: 0.001
```

## API使用

### 1. 使用Tokenizer

```python
from rna_fitness import RNATokenizer

# 初始化
tokenizer = RNATokenizer()

# 编码单个序列
sequence = "AUGCAUGC"
encoded = tokenizer.encode(sequence, max_length=20, return_tensors=True)
print(encoded.shape)  # torch.Size([1, 20])

# 批量编码
sequences = ["AUGC", "UUUU", "GGCC"]
batch = tokenizer.batch_encode(sequences, max_length=20)
print(batch['input_ids'].shape)       # torch.Size([3, 20])
print(batch['attention_mask'].shape)  # torch.Size([3, 20])

# 解码
decoded = tokenizer.decode(encoded[0])
print(decoded)  # "AUGCAUGC"
```

### 2. 使用模型

```python
import torch
from rna_fitness import MambaRNA, RNATokenizer

# 初始化
tokenizer = RNATokenizer()
model = MambaRNA(
    vocab_size=8,
    d_model=256,
    n_layers=4,
    use_mamba=False  # 使用LSTM备选
)

# 编码输入
sequence = "AUGCAUGCAUGC"
batch = tokenizer.batch_encode([sequence], max_length=50)

# 推理
model.eval()
with torch.no_grad():
    predictions = model(batch['input_ids'], batch['attention_mask'])
    print(f"预测fitness: {predictions.item():.4f}")
```

### 3. 使用数据集

```python
from rna_fitness import RNADataset, RNATokenizer
from torch.utils.data import DataLoader

# 准备数据
sequences = ['AUGC' * 10 for _ in range(100)]
fitness = [0.5, 0.8, 0.3, ...]  # 100个值

# 创建数据集
tokenizer = RNATokenizer()
dataset = RNADataset(
    sequences=sequences,
    fitness=fitness,
    tokenizer=tokenizer,
    max_length=100
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 迭代
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    # ... 训练逻辑
```

### 4. 自定义训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from rna_fitness import MambaRNA, RNATokenizer
from rna_fitness.data import create_dataloaders
from rna_fitness.utils.metrics import compute_metrics

# 准备数据
train_sequences = [...]
train_fitness = [...]

# 初始化
tokenizer = RNATokenizer()
model = MambaRNA(vocab_size=8, d_model=256, n_layers=4, use_mamba=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建数据加载器
train_loader, _, _ = create_dataloaders(
    train_sequences=train_sequences,
    train_fitness=train_fitness,
    tokenizer=tokenizer,
    batch_size=32,
    max_length=100
)

# 训练循环
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

## 常见问题

### Q1: 如何使用真正的Mamba SSM？

A: 安装mamba-ssm包，并在配置中设置`use_mamba: true`：

```bash
pip install mamba-ssm causal-conv1d
```

注意：Mamba SSM需要编译，请确保有合适的编译环境和CUDA支持。

### Q2: 训练太慢怎么办？

A: 可以尝试：
1. 减小模型大小（`d_model`, `n_layers`）
2. 增大批次大小（`batch_size`）
3. 使用GPU训练（设置`device.cuda: true`）
4. 减少序列长度（`max_length`）

### Q3: 如何处理不同长度的序列？

A: 框架自动处理不同长度的序列：
- 短序列会被填充到`max_length`
- 长序列会被截断到`max_length`
- attention mask确保模型不关注填充部分

### Q4: 如何解决过拟合？

A: 可以尝试：
1. 增加dropout率（`dropout`）
2. 增加权重衰减（`weight_decay`）
3. 使用早停（`early_stopping_patience`）
4. 收集更多训练数据
5. 使用数据增强

### Q5: 如何选择超参数？

A: 建议的调参顺序：
1. 首先使用默认配置
2. 调整模型大小（`d_model`, `n_layers`）
3. 调整学习率（`learning_rate`）
4. 调整批次大小（`batch_size`）
5. 调整正则化（`dropout`, `weight_decay`）

### Q6: 如何在多GPU上训练？

A: 当前版本支持单GPU训练。多GPU训练需要使用`torch.nn.DataParallel`或`DistributedDataParallel`包装模型。

### Q7: 模型预测结果不理想怎么办？

A: 可以检查：
1. 数据质量和标注正确性
2. 数据量是否足够（建议 > 1000样本）
3. 序列长度设置是否合理
4. 模型是否过拟合或欠拟合
5. 尝试不同的模型架构和超参数

### Q8: 如何保存和加载模型？

A: 训练脚本会自动保存模型到`checkpoints/`目录。手动保存/加载：

```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'my_model.pt')

# 加载
checkpoint = torch.load('my_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 更多资源

- [RNAGym论文和基准](https://github.com/MarksLab-DasLab/RNAGym)
- [Mamba SSM论文](https://arxiv.org/abs/2312.00752)
- [PyTorch文档](https://pytorch.org/docs/)

## 获取帮助

如有问题，请：
1. 查看本文档和README
2. 运行测试确认环境配置正确
3. 在GitHub上提交Issue
