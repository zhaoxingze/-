# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "final_report.docx"

with (RESULTS / "metrics.csv").open("r", encoding="utf-8-sig", newline="") as f:
    ROWS = list(csv.DictReader(f))


def row_for(experiment_id: str, method: str | None = None) -> dict[str, str]:
    for row in ROWS:
        if row["experiment_id"] == experiment_id and (method is None or row["method"] == method):
            return row
    raise KeyError((experiment_id, method))


def pct(row: dict[str, str]) -> float:
    return float(row["test_acc_mean"]) * 100


def trainable(row: dict[str, str]) -> float:
    return float(row["trainable_percent"])


def set_font(run, size=11, bold=False, color=None, name="Times New Roman", east_asia="Microsoft YaHei"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_para(doc: Document, text: str = "", first=True, align=None, size=11):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    if first:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if text:
        set_font(p.add_run(text), size=size)
    return p


def add_heading(doc: Document, text: str, level=1):
    p = doc.add_heading("", level=level)
    color = (46, 116, 181) if level <= 2 else (31, 77, 120)
    set_font(p.add_run(text), 16 if level == 1 else 13, True, color)
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run(text), 10, False, (90, 90, 90))


def shade_cell(cell, fill="F2F4F7"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=8.5):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    set_font(p.add_run(str(text)), size, bold)


def set_table_widths(table, widths_cm):
    table.autofit = False
    for row in table.rows:
        for i, width in enumerate(widths_cm):
            row.cells[i].width = Cm(width)


def add_results_table(doc: Document):
    table = doc.add_table(rows=1, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    headers = ["编号", "方法", "准确率", "标准差", "损失", "可训练参数", "参数比例"]
    widths = [1.0, 2.7, 1.4, 1.3, 1.2, 2.2, 1.5]
    for i, header in enumerate(headers):
        set_cell(table.rows[0].cells[i], header, True)
        shade_cell(table.rows[0].cells[i])
    for row in ROWS:
        cells = table.add_row().cells
        set_cell(cells[0], row["experiment_id"])
        set_cell(cells[1], row["method"])
        set_cell(cells[2], f"{pct(row):.2f}%")
        set_cell(cells[3], f"{float(row['test_acc_std']) * 100:.2f}%")
        set_cell(cells[4], f"{float(row['test_loss_mean']):.4f}")
        set_cell(cells[5], f"{int(row['trainable_params']):,}")
        set_cell(cells[6], f"{trainable(row):.4f}%")
    set_table_widths(table, widths)


def add_image(doc: Document, path: Path, text: str, width=5.85):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, text)


def add_code(doc: Document, code: str, caption: str):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    shade_cell(cell, "F7F7F7")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    for line_no, line in enumerate(code.strip("\n").splitlines()):
        if line_no:
            p.add_run("\n")
        run = p.add_run(line)
        set_font(run, size=8.3, name="Consolas", east_asia="Microsoft YaHei")
    add_caption(doc, caption)


def add_method_table(doc: Document):
    rows = [
        ("E1", "Linear Probing", "冻结骨干，仅训练分类头", "最低成本基准"),
        ("E2", "Partial Fine-tuning", "解冻 layer4 与分类头", "局部解冻基准"),
        ("E3", "Full Fine-tuning", "更新所有参数", "全参数性能基准"),
        ("E4", "SSF-All", "在多层特征后学习 scale/shift", "高效微调核心实验"),
        ("E5", "SSF-L4", "仅调整最高层 layer4 特征", "研究高层特征调整"),
        ("E6", "SSF-L34", "调整 layer3 与 layer4", "研究较少参数下的效果"),
        ("E7", "25% 数据", "主方法在 25% 训练数据上比较", "小样本鲁棒性"),
        ("E8", "50% 数据", "主方法在 50% 训练数据上比较", "数据量影响分析"),
        ("X1", "LoRA", "低秩增量只作用于分类头", "保留扩展与局限分析"),
        ("X2", "Adapter", "输出特征后添加瓶颈适配器", "保留扩展与局限分析"),
    ]
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, header in enumerate(["编号", "方法", "实现方式", "实验目的"]):
        set_cell(table.rows[0].cells[i], header, True)
        shade_cell(table.rows[0].cells[i])
    for item in rows:
        cells = table.add_row().cells
        for i, text in enumerate(item):
            set_cell(cells[i], text, align=WD_ALIGN_PARAGRAPH.LEFT if i > 1 else WD_ALIGN_PARAGRAPH.CENTER)
    set_table_widths(table, [1.0, 2.6, 4.4, 3.7])
    add_caption(doc, "表 1 实验方法、实现方式与目的")


def add_page_number(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("第 "), 9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)
    set_font(footer.add_run(" 页"), 9)


def build():
    e1 = row_for("E1")
    e2 = row_for("E2")
    e3 = row_for("E3")
    e4 = row_for("E4")
    e5 = row_for("E5")
    e6 = row_for("E6")
    x1 = row_for("X1")
    x2 = row_for("X2")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
    add_page_number(sec)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(11)

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("技术报告"), 22, True)
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("基于 ImageNet 预训练 ResNet18 的高效微调方法探索"), 17, True)
    for _ in range(4):
        doc.add_paragraph()
    for label in ["姓名", "学号", "班级", "指导教师"]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(f"{label}: __________________"), 12)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("完成日期: 2026 年 6 月"), 12)
    doc.add_page_break()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("基于 ImageNet 预训练 ResNet18 的高效微调方法探索"), 18, True)
    add_heading(doc, "摘要")
    add_para(
        doc,
        "本文围绕课程设计中“模型微调技术探索”的要求，基于 ImageNet 预训练 ResNet18 与 CIFAR-10 小样本分类任务，"
        "系统比较 Linear Probing、Partial Fine-tuning、Full Fine-tuning 和 SSF-All 四个主体实验，并补充 SSF 层级、"
        "小样本比例、LoRA 与 Adapter 扩展实验。所有配置已统一训练 15 个 epoch，并在两个随机种子上取平均。",
    )
    add_para(
        doc,
        f"实验结果显示，E3 Full Fine-tuning 以 {pct(e3):.2f}% 的平均准确率取得最高性能；"
        f"E4 SSF-All 只训练 {trainable(e4):.4f}% 参数，仍达到 {pct(e4):.2f}% 准确率，体现出较好的参数效率；"
        f"E2 Partial Fine-tuning 准确率为 {pct(e2):.2f}%，说明局部解冻在小样本迁移中也具有较强竞争力。",
    )
    add_para(doc, "关键词：模型微调；参数高效微调；ResNet18；SSF；LoRA；Adapter；CIFAR-10", False)
    doc.add_page_break()

    add_heading(doc, "1 引言")
    add_para(
        doc,
        "深度神经网络在视觉识别任务中通常依赖大量标注数据。对于课程实验或实际应用中的小样本场景，"
        "直接从零训练模型既不经济，也容易出现欠拟合或过拟合。预训练模型迁移学习能够复用大规模数据上学到的通用视觉表示，"
        "因此成为图像分类实验中最常见的起点。",
    )
    add_para(
        doc,
        "传统微调方式通常包括只训练分类头、局部解冻和全参数微调。它们实现简单、可解释性强，适合作为基准；"
        "但当模型规模进一步扩大时，全量更新会带来训练显存、存储和部署成本。参数高效微调方法试图只更新极少量任务相关参数，"
        "在保持性能的同时降低资源需求。",
    )
    add_para(
        doc,
        "本项目选择 ResNet18 作为骨干网络，一方面因为它结构清晰、训练速度较快；另一方面也便于在卷积网络上实现 SSF、Adapter "
        "和简化 LoRA 等思想。实验目标不是追求 CIFAR-10 的最高绝对准确率，而是比较不同微调策略在相同数据、轮数和随机种子下的相对表现。",
    )
    doc.add_page_break()

    add_heading(doc, "2 实验设计")
    add_para(
        doc,
        "实验列表按照课程建议划分为必做实验与加分实验。E1 至 E4 构成主体比较：E1 提供最低成本基线，E2 提供局部解冻基线，"
        "E3 提供全参数性能基线，E4 检验 SSF 在卷积骨干中的参数高效微调效果。E5 至 E8 用于观察 SSF 层级和训练数据量变化的影响。",
    )
    add_method_table(doc)
    add_para(
        doc,
        "保留实验 LoRA 与 Adapter 用于扩展讨论。需要强调的是，本项目中的 LoRA 只作用于新初始化分类头，没有插入到预训练骨干层，"
        "因此它只能代表一种简化实现。Adapter 则在 ResNet18 输出特征后增加瓶颈模块，通过残差修正改善特征表达。",
    )
    doc.add_page_break()

    add_heading(doc, "3 数据集与训练设置")
    add_para(
        doc,
        "数据集采用 CIFAR-10。为了控制计算成本，完整数据设置中每个类别抽取 100 张训练图像与 100 张测试图像，"
        "E7 和 E8 分别把训练集比例缩小为 25% 与 50%。所有图像缩放到 128x128，并使用 ImageNet 均值和标准差归一化，"
        "以匹配预训练 ResNet18 的输入分布。",
    )
    add_para(
        doc,
        "训练轮数统一为 15 个 epoch，随机种子为 2026 与 2027。优化器使用 AdamW；除 Full Fine-tuning 使用较小学习率外，"
        "其余方法主要训练新增层或少量参数，因此使用较大的 head learning rate。训练时启用 CUDA 混合精度以缩短实验时间。",
    )
    add_code(
        doc,
        """
training:
  epochs: 15
  batch_size: 128
  lr_head: 0.002
  lr_full: 0.00005
  weight_decay: 0.0001
  seeds: [2026, 2027]
""",
        "代码片段 1 统一后的训练配置",
    )
    doc.add_page_break()

    add_heading(doc, "4 关键实现")
    add_para(
        doc,
        "训练入口 `train.py` 负责读取配置、构建模型、训练和汇总结果。为了满足训练过程可观察性要求，本次修改在每个方法、每个随机种子上增加外层进度条，"
        "并在每个 epoch 内增加 batch 级进度条。进度条显示当前实验编号、方法名、seed、epoch 和最新 loss，训练结束后仍保留结果行。",
    )
    add_code(
        doc,
        """
epoch_iter = tqdm(
    range(1, epochs + 1),
    desc=f"{config['experiment_id']} {method} seed={seed}",
    leave=False,
)
for epoch in epoch_iter:
    batch_iter = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
    for x, y in batch_iter:
        ...
        batch_iter.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
    epoch_iter.set_postfix(test_acc=f"{test_acc:.4f}")
""",
        "代码片段 2 训练进度条实现",
    )
    add_para(
        doc,
        "模型构建函数根据 `method` 字段选择不同训练策略。对于普通微调方法，骨干网络来自 `torchvision.models.resnet18` 的 ImageNet 预训练权重；"
        "对于 SSF 方法，则在指定 stage 后加入可学习的 scale 和 shift 参数。",
    )
    add_heading(doc, "5 微调模型实现")
    add_para(
        doc,
        "Linear Probing 冻结所有骨干参数，仅训练最后的线性分类头。Partial Fine-tuning 解冻 layer4 与分类头，在计算成本和表示调整能力之间折中。"
        "Full Fine-tuning 更新全部参数，理论表达能力最强，但在小样本下更容易受到数据规模限制。",
    )
    add_code(
        doc,
        """
if method == "linear_probe":
    for param in model.classifier.parameters():
        param.requires_grad = True
elif method == "partial_ft":
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
""",
        "代码片段 3 Linear Probing 与 Partial Fine-tuning 的可训练参数选择",
    )
    add_para(
        doc,
        "SSF 模块对特征做逐通道缩放和平移。与直接更新卷积核相比，SSF 的新增参数非常少，却能对预训练特征分布进行任务相关调整。"
        "本项目实现了 SSF-All、SSF-L4 和 SSF-L34 三种配置，用于比较不同作用层级的影响。",
    )
    doc.add_page_break()

    add_heading(doc, "6 LoRA 与 Adapter 扩展")
    add_para(
        doc,
        "LoRA 的核心思想是在原权重旁加入低秩增量矩阵，使任务更新可以由少量参数表达。本项目保留了 LoRA 实验，"
        "但它只作用于新初始化分类头，而不是预训练骨干层。因此该实验更适合用于分析“简化实现的局限”，不能直接代表原始 LoRA 在大型预训练模型中的效果。",
    )
    add_code(
        doc,
        """
delta = self.b @ self.a * self.scaling
return F.linear(x, self.weight + delta, self.bias)
""",
        "代码片段 4 分类头 LoRA 的低秩增量",
    )
    add_para(
        doc,
        "Adapter 借鉴瓶颈适配器思想，在 ResNet18 输出特征后添加 `Linear-ReLU-Linear` 模块，并通过残差连接修正特征。"
        "这种实现新增参数少、结构直观，但由于插入位置靠后，能够调整的表示空间有限。",
    )
    add_code(
        doc,
        """
z = self.backbone(x)
z = z + self.adapter(z)
return self.classifier(z)
""",
        "代码片段 5 输出特征后的瓶颈 Adapter 残差修正",
    )
    doc.add_page_break()

    add_heading(doc, "7 实验结果")
    add_para(
        doc,
        "表 2 汇总了所有实验在两个随机种子上的平均结果。准确率用于衡量分类性能，标准差用于观察稳定性，"
        "可训练参数比例用于衡量参数效率。由于训练数据量较小，绝对准确率会受到采样和随机初始化影响，因此分析重点放在方法之间的相对差异。",
    )
    add_results_table(doc)
    add_caption(doc, "表 2 15 轮训练后的实验结果汇总")
    doc.add_page_break()

    add_heading(doc, "8 准确率对比")
    add_image(doc, RESULTS / "accuracy_comparison.png", "图 1 不同方法的测试准确率对比", width=6.0)
    add_para(
        doc,
        f"从准确率看，E3 Full Fine-tuning 达到 {pct(e3):.2f}%，是完整数据设置下的最高结果；"
        f"E2 Partial Fine-tuning 为 {pct(e2):.2f}%，与全参数微调差距较小；"
        f"E4 SSF-All 为 {pct(e4):.2f}%，在几乎不更新骨干参数的情况下保持了较强性能。",
    )
    doc.add_page_break()

    add_heading(doc, "9 参数效率对比")
    add_image(doc, RESULTS / "parameter_comparison.png", "图 2 可训练参数比例对比", width=6.0)
    add_para(
        doc,
        f"参数效率方面，E1 只训练 {trainable(e1):.4f}% 参数，成本最低；E4 只训练 {trainable(e4):.4f}% 参数，"
        f"但准确率比 E1 高 {pct(e4) - pct(e1):.2f} 个百分点。E2 与 E3 的参数比例分别为 {trainable(e2):.4f}% 和 {trainable(e3):.1f}%，"
        "说明它们的性能收益来自更大范围的骨干更新。",
    )
    doc.add_page_break()

    add_heading(doc, "10 训练曲线分析")
    add_image(doc, RESULTS / "training_curves.png", "图 3 不同方法的测试准确率训练曲线", width=6.0)
    add_para(
        doc,
        "训练曲线显示，大部分方法在前几个 epoch 快速提升，随后进入较缓慢的震荡阶段。Linear Probing 收敛较快，说明预训练特征已经较适合分类；"
        "Full Fine-tuning 与 Partial Fine-tuning 在后期仍有提升空间，但也更容易受小样本随机性影响。SSF 方法曲线较稳定，符合少量参数调整的预期。",
    )
    doc.add_page_break()

    add_heading(doc, "11 小样本与 SSF 层级分析")
    e7_lp = row_for("E7", "linear_probe")
    e7_pf = row_for("E7", "partial_ft")
    e7_ff = row_for("E7", "full_ft")
    e7_ssf = row_for("E7", "ssf_all")
    e8_lp = row_for("E8", "linear_probe")
    e8_pf = row_for("E8", "partial_ft")
    e8_ff = row_for("E8", "full_ft")
    e8_ssf = row_for("E8", "ssf_all")
    add_para(
        doc,
        f"在 25% 数据设置 E7 中，Partial Fine-tuning 达到 {pct(e7_pf):.2f}%，SSF-All 达到 {pct(e7_ssf):.2f}%，"
        f"Linear Probing 为 {pct(e7_lp):.2f}%，Full Fine-tuning 为 {pct(e7_ff):.2f}%。"
        "这说明数据极少时，局部解冻仍然能提供较强的适应能力，而全参数更新的优势会被样本不足削弱。",
    )
    add_para(
        doc,
        f"在 50% 数据设置 E8 中，Full Fine-tuning 达到 {pct(e8_ff):.2f}%，Partial Fine-tuning 为 {pct(e8_pf):.2f}%，"
        f"SSF-All 为 {pct(e8_ssf):.2f}%，Linear Probing 为 {pct(e8_lp):.2f}%。随着数据量增加，全参数更新重新获得优势，"
        "但 SSF 与局部解冻仍保持接近表现。",
    )
    add_para(
        doc,
        f"SSF 层级实验中，SSF-L4 准确率为 {pct(e5):.2f}%，SSF-L34 为 {pct(e6):.2f}%，SSF-All 为 {pct(e4):.2f}%。"
        "结果说明只调整最高层特征已经有效，进一步加入 layer3 可提升表现，而所有层都加入 SSF 时性能最高。",
    )
    add_heading(doc, "12 局限性与改进方向")
    add_para(
        doc,
        f"LoRA 扩展实验的准确率为 {pct(x1):.2f}%，明显低于主体方法。主要原因是本实现只在分类头上使用低秩增量，"
        "没有对预训练骨干中的卷积或线性映射进行低秩更新；分类头又是新初始化层，因此 LoRA 的优势没有充分体现。",
    )
    add_para(
        doc,
        f"Adapter 实验达到 {pct(x2):.2f}%，可训练参数比例为 {trainable(x2):.4f}%。它比 Linear Probing 增加了少量特征修正能力，"
        "但由于插入位置在最终输出特征之后，无法逐层调整中间视觉表示。后续可以把 Adapter 插入 ResNet block 内部，并比较不同瓶颈维度。",
    )
    add_para(
        doc,
        "本实验为了保证课程设计可复现性，采用了较小的每类样本数和两个随机种子。若计算资源允许，可以扩大训练集、增加随机种子、"
        "加入验证集选择最佳 epoch，并把 LoRA/Adapter 的插入位置扩展到骨干网络内部，从而获得更稳健的结论。",
    )
    doc.add_page_break()

    add_heading(doc, "13 结论")
    add_para(
        doc,
        "本文在统一 15 轮训练设置下完成了 Linear Probing、Partial Fine-tuning、Full Fine-tuning、SSF-All、SSF 层级实验、"
        "小样本实验以及 LoRA/Adapter 扩展实验。结果表明，全参数微调在当前完整数据设置下取得最高准确率，"
        "但参数开销最大；Partial Fine-tuning 兼顾性能和实现简洁性；SSF-All 在极低可训练参数比例下获得接近局部解冻的性能，"
        "最能体现参数高效微调的核心价值。",
    )
    add_para(
        doc,
        "从课程设计角度看，E1 至 E4 已构成完整主体实验闭环，E5 至 E8 进一步解释了层级选择和数据量变化的影响。"
        "保留的 LoRA 与 Adapter 实验也提供了方法局限分析：参数高效微调不仅取决于参数量，还取决于插入位置与预训练表示之间的关系。",
    )
    add_heading(doc, "参考文献")
    refs = [
        "He K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.",
        "Lian D. et al. Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning. NeurIPS, 2022.",
        "Hu E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.",
        "Chen S. et al. AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition. NeurIPS, 2022.",
        "Ben Zaken E. et al. BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models. ACL, 2022.",
    ]
    for i, ref in enumerate(refs, 1):
        add_para(doc, f"[{i}] {ref}", False)

    doc.save(REPORT)
    print(REPORT, REPORT.stat().st_size)


if __name__ == "__main__":
    build()
