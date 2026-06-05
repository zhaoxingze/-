# 深度学习模型微调技术探索

本项目对应 `课程设计-v1.pptx` 中的课程设计题目“模型微调技术探索”。项目以 ImageNet 预训练 ResNet18 为骨干网络，在 CIFAR-10 小样本图像分类任务上比较多种微调策略，并保留 LoRA 与 Adapter 的简化实现用于扩展分析。

## 项目结构

```text
efficient_finetuning_project/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── dataset.py
├── utils.py
├── configs/
│   ├── linear_probe.yaml
│   ├── partial_ft.yaml
│   ├── full_ft.yaml
│   ├── ssf.yaml
│   ├── ssf_l4.yaml
│   ├── ssf_l34.yaml
│   ├── fewshot_25.yaml
│   ├── fewshot_50.yaml
│   ├── lora.yaml
│   └── adapter.yaml
├── models/
│   ├── resnet_finetune.py
│   └── ssf_resnet.py
├── experiments/
├── results/
│   ├── metrics.csv
│   ├── accuracy_comparison.png
│   ├── parameter_comparison.png
│   └── training_curves.png
└── report/
    └── final_report.docx
```

## 实验环境

推荐环境如下：

- Python 3.10 或以上
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PyYAML
- python-docx
- tqdm
- CUDA GPU，本次实验设备为 `NVIDIA GeForce RTX 4060 Laptop GPU`

安装依赖：

```powershell
pip install -r requirements.txt
```

如果 Windows 环境中导入 PyTorch/torchvision 时出现 OpenMP 重复加载提示，项目入口会设置 `KMP_DUPLICATE_LIB_OK=TRUE`，用于保证课程实验脚本可运行。

## 数据集下载

本实验使用 CIFAR-10 数据集和 ImageNet 预训练 ResNet18 权重。数据目录约定：

```text
data/
└── cifar-10-python.tar.gz

models/
└── torch/hub/checkpoints/resnet18-f37072fd.pth
```

默认配置中 `download: false`，表示运行前需要本地已有 CIFAR-10 和 ResNet18 权重。若 `data/cifar-10-batches-py` 解压目录不可读，`dataset.py` 会自动从 `data/cifar-10-python.tar.gz` 读取数据。

## 实验列表

### 主体实验

| 编号 | 方法 | 目的 | 配置文件 |
|---|---|---|---|
| E1 | Linear Probing | 最低成本基准 | `configs/linear_probe.yaml` |
| E2 | Partial Fine-tuning | 局部解冻基准 | `configs/partial_ft.yaml` |
| E3 | Full Fine-tuning | 全参数性能基准 | `configs/full_ft.yaml` |
| E4 | SSF-All | 高效微调核心实验 | `configs/ssf.yaml` |

| 编号 | 方法 | 目的 | 配置文件 |
|---|---|---|---|
| E5 | SSF-L4 | 研究只调整最高层特征 | `configs/ssf_l4.yaml` |
| E6 | SSF-L34 | 研究较少参数下的效果 | `configs/ssf_l34.yaml` |
| E7 | 小样本 25% 数据训练 | 比较不同方法在更少数据下的表现 | `configs/fewshot_25.yaml` |
| E8 | 小样本 50% 数据训练 | 分析数据量变化影响 | `configs/fewshot_50.yaml` |

#### LoRA

在分类头上使用低秩增量矩阵。作用：对应 LoRA 的低秩更新思想。但本实验的 LoRA 只作用于新初始化分类头，没有作用于预训练骨干层，因此结果需要作为简化实现的局限来分析。

#### Adapter

在 ResNet18 输出特征后添加瓶颈适配器。作用：借鉴 AdaptFormer/Adapter 思想，用少量新增参数对特征进行残差修正。

## 运行方式

运行单个实验：

```powershell
python train.py --config configs/ssf.yaml
```

运行全部配置：

```powershell
python train.py --all
```

查看已保存的汇总结果：

```powershell
python evaluate.py --metrics results/metrics.csv
```

训练完成后会生成：

- `results/history.csv`：每个 epoch 的测试损失和准确率
- `results/seed_results.csv`：每个 seed 的最终结果
- `results/metrics.csv`：跨 seed 汇总结果
- `results/accuracy_comparison.png`：准确率对比图
- `results/parameter_comparison.png`：可训练参数比例对比图
- `results/training_curves.png`：训练曲线图

## 实验结果

新一轮 CUDA 实验使用每类 100 张训练图像、100 张测试图像，统一训练 15 个 epoch，并重复两个随机种子。结果位于 `results/`。

| 方法 | 平均测试准确率 | 可训练参数比例 | 说明 |
|---|---:|---:|---|
| E1 Linear Probing | 74.30% | 0.0459% | 只训练分类头，作为最低成本基准 |
| E2 Partial Fine-tuning | 78.55% | 75.1129% | 解冻 layer4 和分类头，作为局部解冻基准 |
| E3 Full Fine-tuning | 80.85% | 100.0000% | 全参数微调基准 |
| E4 SSF-All | 78.40% | 0.0722% | 在多个 stage 后加入 scale/shift，是核心高效微调实验 |
| E5 SSF-L4 | 74.75% | 0.0642% | 只调整最高层特征 |
| E6 SSF-L34 | 76.85% | 0.0688% | 调整 layer3 和 layer4 |
| X1 LoRA | 35.60% | 0.0374% | 只作用于新分类头，因此作为简化实现局限分析 |
| X2 Adapter | 71.35% | 0.6334% | 在输出特征后添加瓶颈适配器 |

结论：E3 Full Fine-tuning 在当前完整数据设置下准确率最高；E4 SSF-All 只训练 0.0722% 参数，准确率达到 78.40%，更能体现参数高效微调的核心价值。LoRA 的低结果主要来自简化实现位置，不应直接解释为原始 LoRA 方法无效。

## 报告

课程报告按 PPT 要求采用小论文形式，内容包含摘要、引言、方法、关键实现、实验结果、分析、结论和参考文献。最终 Word 小论文位于 `report/final_report.docx`，当前版本已扩展至 12-13 页，并包含若干关键代码片段。

## 参考文献

1. He K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
2. Lian D. et al. Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning. NeurIPS, 2022.
3. Hu E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.
4. Chen S. et al. AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition. NeurIPS, 2022.
5. Ben Zaken E. et al. BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models. ACL, 2022.
