# 深度学习模型微调技术探索

本项目按照 `课程设计-v1.pptx` 中第四个具体选题“模型微调技术探索”完成，参考 AdaptFormer、SSF、LoRA 和 BitFit 四篇论文，使用 CUDA 跑通参数高效微调对比实验，并生成课程设计小论文。

## 主要交付

- `模型微调技术探索_课程设计小论文.docx`：最终小论文。
- `完成步骤.md`：任务划分、执行步骤和实验结论记录。
- `src/run_finetuning_cuda_experiment.py`：CUDA 微调实验脚本。
- `outputs/finetune/`：实验结果、逐 seed 记录和准确率曲线。
- `materials/finetune_pdfs/`：4 篇参考 PDF 的文本提取结果。
- `tools/build_finetune_report_docx.py`：根据实验结果生成 Word 小论文的脚本。

## CUDA 实验摘要

实验使用 ImageNet 预训练 ResNet18 和 CIFAR-10 小样本设置，每类 100 张训练图像、100 张测试图像，在 `NVIDIA GeForce RTX 4060 Laptop GPU` 上运行。

| 方法 | 平均测试准确率 | 可训练参数比例 |
|---|---:|---:|
| LINEAR | 71.25% | 0.0459% |
| BITFIT | 77.25% | 0.0888% |
| SSF | 71.20% | 0.0550% |
| LORA | 14.25% | 0.0374% |
| ADAPTER | 71.30% | 0.6334% |
| FULL | 72.80% | 100.0000% |

结论：在本小样本迁移设置下，BitFit 以极低的可训练参数比例取得最高准确率；LoRA 的低结果主要来自本实验将其简化应用于新初始化分类头，而不是原论文中的预训练层低秩增量。

## 未入库内容

以下文件可重新下载或生成，且体积较大，因此通过 `.gitignore` 排除：

- `data/`：CIFAR-10 数据集。
- `models/`：ImageNet 预训练 ResNet18 权重。
- `outputs/tmp/`：DOCX 渲染临时目录。

另外，前期误选“新型优化算法”方向时生成的 SAM/ESAM 脚本、材料和输出没有纳入本次提交，以保持仓库聚焦在第四题“模型微调技术探索”。
