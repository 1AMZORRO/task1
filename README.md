# RNA Fitness 预测项目

基于 Mamba SSM 的 RNA 适应性（Fitness）预测模型。本项目实现了一个简洁高效的深度学习系统，用于预测RNA序列的适应性得分。

## 🌟 项目简介

本项目使用 **Mamba 状态空间模型（SSM）** 作为核心架构，实现对 RNA 序列适应性的预测。项目参照 [RNAGym](https://github.com/MarksLab-DasLab/RNAGym) 的任务定义、数据集和评分标准。

### 为什么选择 Mamba SSM？

- ✅ **长序列建模能力强**：Mamba SSM 能够高效处理长RNA序列
- ✅ **计算效率高**：相比Transformer，计算复杂度更低
- ✅ **位置敏感性好**：对序列位置信息的捕获能力强

## ✨ 特性

- ✅ 基于 Mamba SSM 的深度学习模型
- ✅ 支持 GPU 加速训练
- ✅ 完整的训练和验证流程
- ✅ 多种评估指标：Spearman 相关系数、AUC、MCC
- ✅ 训练过程可视化
- ✅ 预测结果可视化
- ✅ 简化版本支持（无需GPU也可运行）

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

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如果无法安装 `mamba-ssm`（需要CUDA支持），项目会自动使用简化版本（基于GRU），可以在CPU上运行。

### 2. 运行快速测试

```bash
# 测试所有模块是否正常工作
python test_quick.py
```

### 3. 训练模型

```bash
# 基础训练（使用默认参数）
python train.py --epochs 50

# 自定义训练参数
python train.py \
    --dataset Andreasson_2020_ribozyme \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --d_model 256 \
    --n_layers 4 \
    --output_dir outputs
```

### 4. 评估模型

```bash
python evaluate.py \
    --model_path outputs/best_model.pt \
    --dataset Andreasson_2020_ribozyme \
    --output_dir outputs
```

## 📊 训练结果示例

训练完成后会在输出目录生成以下文件：

1. **训练曲线** (`training_curves.png`)
   - 损失曲线
   - Spearman 相关系数变化
   - AUC、MCC 分数变化
   - RMSE、MAE 误差变化

2. **预测散点图** (`prediction_scatter.png`)
   - 预测值 vs 真实值对比
   - 评估指标展示

3. **结果摘要** (`results_summary.txt`)
   - 所有评估指标的详细数值

4. **最佳模型** (`best_model.pt`)
   - 验证集表现最好的模型检查点

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

## 📈 评估指标说明

本项目使用以下评估指标：

- **Spearman 相关系数** 📊
  - 衡量预测值与真实值的单调相关性
  - 范围：[-1, 1]，越接近1表示相关性越强

- **AUC (Area Under ROC Curve)** 📈
  - 二分类性能指标，使用中位数作为阈值
  - 范围：[0, 1]，越接近1表示分类性能越好

- **MCC (Matthews Correlation Coefficient)** ⚖️
  - 二分类质量的综合指标
  - 范围：[-1, 1]，越接近1表示预测质量越好

- **RMSE (Root Mean Square Error)** 📉
  - 回归任务的均方根误差
  - 越小表示预测越准确

- **MAE (Mean Absolute Error)** 📉
  - 平均绝对误差
  - 越小表示预测越准确

## 🏗️ 模型架构

模型采用 Mamba SSM 作为核心模块，架构如下：

```
输入 RNA 序列 (AUGC...)
    ↓
Token 嵌入层 (Embedding)
    ↓
Mamba SSM 层 × N
    ├─ 状态空间模型
    ├─ LayerNorm
    └─ 残差连接
    ↓
LayerNorm + Pooling
    ↓
回归头 (Regression Head)
    ↓
Fitness 分数预测
```

### Mamba SSM 的优势

1. **线性复杂度**：相比 Transformer 的 O(n²)，Mamba 的复杂度为 O(n)
2. **长序列处理**：能够高效处理长达数百个碱基的 RNA 序列
3. **状态记忆**：通过状态空间模型保持序列的长期依赖关系

## 💻 GPU 使用

训练脚本会自动检测并使用可用的 GPU。

**查看 GPU 使用情况：**
```bash
nvidia-smi
```

**如果没有 GPU**：项目会自动使用简化版 Mamba 实现，可以在 CPU 上运行（速度较慢）。

## 📝 可用数据集

项目包含来自 RNAGym 的 30 个数据集，涵盖不同类型的 RNA：

- **核酶 (Ribozyme)**：如 Andreasson_2020、Beck_2022 等
- **tRNA**：如 Domingo_2018、Guy_2014 等
- **mRNA**：如 Julien_2016、Ke_2017
- **适配体 (Aptamer)**：如 Townshend_2015 系列

运行训练时会显示所有可用数据集。

## ⚠️ 注意事项

1. **首次安装 mamba-ssm**：可能需要编译，请确保安装了正确的 CUDA 版本（如果有GPU）
2. **内存不足**：可以减小 `--batch_size` 或 `--d_model` 参数
3. **序列长度**：不同数据集的序列长度不同，可能需要调整 `--max_length` 参数
4. **训练时间**：在 GPU 上训练 50 epochs 大约需要 10-30 分钟（取决于数据集大小）
5. **中文显示**：可视化图表中的中文可能无法正常显示（缺少中文字体），但不影响功能

## 🔧 故障排除

**问题：无法安装 mamba-ssm**
```bash
# 解决方案：项目会自动使用简化版本（基于GRU）
# 或者尝试手动安装：
pip install mamba-ssm --no-build-isolation
```

**问题：CUDA out of memory**
```bash
# 解决方案：减小批次大小和模型维度
python train.py --batch_size 16 --d_model 128
```

**问题：训练速度慢**
```bash
# 解决方案：减少层数，使用更小的模型
python train.py --n_layers 2 --d_model 128
```

## 📚 参考文献

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [RNAGym: A Platform for RNA Fitness Prediction](https://github.com/MarksLab-DasLab/RNAGym)

## 📄 许可

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请提交 GitHub Issue。

---

**项目特点**：
- ✨ 简洁易用的代码结构
- 📊 完整的评估指标体系
- 🎨 精美的可视化结果
- 🚀 支持 GPU 加速
- 📖 详细的文档说明
