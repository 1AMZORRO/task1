# RNA Fitness预测框架

基于Mamba SSM模型的RNA fitness预测框架，参照[RNAGym](https://github.com/MarksLab-DasLab/RNAGym)标准实现。

## 项目简介

本项目实现了一个用于RNA适应性（fitness）预测的深度学习框架，核心模型采用Mamba状态空间模型（SSM）。Mamba模型是一种高效的序列建模架构，特别适合处理RNA这类长序列数据。

### 主要特性

- ✅ **Mamba SSM模型**: 使用先进的状态空间模型进行序列建模
- ✅ **RNAGym标准**: 遵循RNAGym的任务定义和评估指标
- ✅ **完整的训练流程**: 包含数据处理、模型训练、评估等完整pipeline
- ✅ **灵活的配置**: 通过YAML配置文件轻松调整模型和训练参数
- ✅ **详细的评估指标**: 提供MSE、R²、Spearman相关系数等多种评估指标

## 项目结构

```
task1/
├── rna_fitness/              # 核心模块
│   ├── models/               # 模型实现
│   │   ├── __init__.py
│   │   └── mamba_rna.py     # Mamba RNA模型
│   ├── data/                 # 数据处理
│   │   ├── __init__.py
│   │   └── dataset.py       # 数据集类
│   ├── utils/                # 工具函数
│   │   ├── __init__.py
│   │   ├── tokenizer.py     # RNA序列tokenizer
│   │   └── metrics.py       # 评估指标
│   └── configs/              # 配置文件
│       └── default_config.yaml
├── examples/                 # 示例代码
│   └── quick_start.py       # 快速开始示例
├── tests/                    # 测试代码
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_dataset.py
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── requirements.txt          # 依赖
├── setup.py                  # 安装配置
└── README.md                 # 本文件
```

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/1AMZORRO/task1.git
cd task1

# 安装依赖
pip install -r requirements.txt

# 安装本项目
pip install -e .
```

### Mamba SSM安装（可选）

如果要使用真正的Mamba模型（而不是LSTM备选），需要额外安装：

```bash
pip install mamba-ssm causal-conv1d
```

注意：Mamba SSM的安装可能需要编译，请确保有合适的编译环境。

## 快速开始

### 1. 运行示例代码

```bash
python examples/quick_start.py
```

这将展示：
- RNA序列的编码和解码
- 模型推理
- 数据集的使用
- 简单的训练循环

### 2. 训练模型

使用示例数据训练：

```bash
python train.py --config rna_fitness/configs/default_config.yaml
```

使用自己的数据训练：

```bash
python train.py \
    --config rna_fitness/configs/default_config.yaml \
    --train_data path/to/train.csv \
    --val_data path/to/val.csv \
    --test_data path/to/test.csv
```

数据格式要求（CSV文件）：
```csv
sequence,fitness
AUGCAUGCAUGC,0.5
UUUUAAAACCCC,0.8
GGGGUUUUCCCC,0.3
```

### 3. 评估模型

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data path/to/test.csv \
    --output predictions.csv
```

## 配置说明

配置文件位于`rna_fitness/configs/default_config.yaml`，主要配置项：

### 模型配置
```yaml
model:
  vocab_size: 8      # 词汇表大小
  d_model: 256       # 模型维度
  n_layers: 4        # 层数
  d_state: 16        # 状态空间维度
  dropout: 0.1       # Dropout率
  use_mamba: true    # 是否使用Mamba（false使用LSTM）
```

### 训练配置
```yaml
training:
  num_epochs: 100              # 训练轮数
  learning_rate: 0.001         # 学习率
  batch_size: 32               # 批次大小
  early_stopping_patience: 10  # 早停耐心值
```

## API使用

### Tokenizer

```python
from rna_fitness import RNATokenizer

# 初始化
tokenizer = RNATokenizer()

# 编码单个序列
encoded = tokenizer.encode("AUGCAUGC", max_length=20)

# 批量编码
batch = tokenizer.batch_encode(["AUGC", "UUUU"], max_length=20)

# 解码
decoded = tokenizer.decode(encoded[0])
```

### 模型

```python
from rna_fitness import MambaRNA
import torch

# 初始化模型
model = MambaRNA(
    vocab_size=8,
    d_model=256,
    n_layers=4
)

# 推理
input_ids = torch.randint(0, 8, (2, 20))
attention_mask = torch.ones(2, 20)
predictions = model(input_ids, attention_mask)
```

### 数据集

```python
from rna_fitness import RNADataset, RNATokenizer

tokenizer = RNATokenizer()

# 创建数据集
dataset = RNADataset(
    sequences=['AUGC', 'UUUU'],
    fitness=[0.5, 0.8],
    tokenizer=tokenizer,
    max_length=20
)

# 获取样本
sample = dataset[0]
```

## 评估指标

根据RNAGym标准，本框架提供以下评估指标：

### RNAGym标准指标（主要）

- **Spearman相关系数（绝对值）**: 秩相关系数，用于衡量预测值和真实值的单调关系
- **AUC**: 基于中位数阈值的ROC曲线下面积，评估二分类性能
- **MCC（绝对值）**: Matthews相关系数，评估二分类预测质量

### 额外提供的指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **R²**: 决定系数
- **Pearson相关系数**: 线性相关系数

这些指标与RNAGym论文中使用的评估标准完全一致。

## 测试

运行测试：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_tokenizer.py -v
pytest tests/test_model.py -v
pytest tests/test_dataset.py -v
```

## 参考

- [RNAGym](https://github.com/MarksLab-DasLab/RNAGym): RNA适应性预测基准
- [Mamba SSM](https://github.com/state-spaces/mamba): 状态空间模型

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过Issue联系。
