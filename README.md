# RNA Fitness 预测项目

基于 Mamba SSM 的 RNA 适应性（Fitness）预测模型。

## 项目简介

本项目使用 Mamba 状态空间模型（SSM）作为核心架构，实现对 RNA 序列适应性的预测。项目参照 [RNAGym](https://github.com/MarksLab-DasLab/RNAGym) 的任务定义、数据集和评分标准。

## 特性

- ✅ 基于 Mamba SSM 的深度学习模型
- ✅ 支持 GPU 加速训练
- ✅ 完整的训练和验证流程
- ✅ 多种评估指标：Spearman 相关系数、AUC、MCC
- ✅ 训练过程可视化
- ✅ 预测结果可视化

## 项目结构

```
.
├── data/
│   └── RNAGym/          # RNA 数据集（CSV 格式）
├── models/
│   └── mamba_rna.py     # Mamba SSM 模型定义
├── utils/
│   ├── data_loader.py   # 数据加载和预处理
│   ├── metrics.py       # 评估指标计算
│   └── visualization.py # 可视化工具
├── train.py             # 训练脚本
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- mamba-ssm >= 1.0.0
- scikit-learn
- pandas
- matplotlib

## 数据集

项目使用 RNAGym 数据集，数据格式为 CSV 文件，包含以下列：
- `mutant`: 突变信息
- `DMS_score`: 适应性得分（目标值）
- `sequence`: RNA 序列

可用数据集位于 `data/RNAGym/` 目录下。

## 使用方法

### 基础训练

```bash
python train.py --dataset Andreasson_2020_ribozyme --epochs 50
```

### 自定义参数训练

```bash
python train.py \
    --dataset Andreasson_2020_ribozyme \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --d_model 256 \
    --n_layers 4 \
    --output_dir outputs
```

### 主要参数说明

**数据参数：**
- `--data_dir`: 数据目录路径（默认：`data/RNAGym`）
- `--dataset`: 数据集名称（默认：`Andreasson_2020_ribozyme`）
- `--max_length`: 最大序列长度（默认：512）
- `--train_ratio`: 训练集比例（默认：0.8）

**模型参数：**
- `--d_model`: 模型维度（默认：256）
- `--n_layers`: Mamba 层数（默认：4）
- `--d_state`: SSM 状态维度（默认：16）
- `--dropout`: Dropout 率（默认：0.1）

**训练参数：**
- `--batch_size`: 批次大小（默认：32）
- `--epochs`: 训练轮数（默认：50）
- `--learning_rate`: 学习率（默认：1e-4）
- `--output_dir`: 输出目录（默认：`outputs`）

## 输出结果

训练完成后，会在输出目录中生成以下文件：

1. **best_model.pt**: 最佳模型检查点
2. **training_curves.png**: 训练曲线图
   - 损失曲线
   - Spearman 相关系数
   - AUC 分数
   - MCC 分数
   - RMSE 和 MAE
3. **prediction_scatter.png**: 预测值 vs 真实值散点图
4. **results_summary.txt**: 评估结果摘要

## 评估指标

- **Spearman 相关系数**: 衡量预测值与真实值的单调相关性
- **AUC (Area Under ROC Curve)**: 二分类性能指标
- **MCC (Matthews Correlation Coefficient)**: 二分类质量指标
- **RMSE (Root Mean Square Error)**: 回归误差
- **MAE (Mean Absolute Error)**: 平均绝对误差

## 模型架构

模型采用 Mamba SSM 作为核心模块：

1. **Token 嵌入层**: 将 RNA 碱基（A, U, G, C）转换为向量表示
2. **Mamba 层堆叠**: 多层 Mamba SSM 块，捕获序列依赖关系
3. **回归头**: 预测适应性得分

Mamba SSM 的优势：
- 长序列建模能力强
- 计算效率高于 Transformer
- 对位置信息敏感

## GPU 使用

训练脚本会自动检测并使用可用的 GPU。如果没有 GPU，会自动使用 CPU。

查看 GPU 使用情况：
```bash
nvidia-smi
```

## 示例：快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型（使用默认数据集）
python train.py --epochs 30 --batch_size 32

# 3. 查看结果
ls outputs/
```

## 注意事项

1. 首次安装 `mamba-ssm` 可能需要编译，请确保安装了正确的 CUDA 版本
2. 训练时建议使用 GPU 以加快速度
3. 不同数据集的序列长度不同，可能需要调整 `--max_length` 参数
4. 如果遇到内存不足，可以减小 `--batch_size` 或 `--d_model`

## 参考

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [RNAGym: A Platform for RNA Fitness Prediction](https://github.com/MarksLab-DasLab/RNAGym)

## 许可

MIT License
