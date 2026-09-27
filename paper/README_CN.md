# 论文使用说明

`main.tex` 是英文 IEEE TPAMI 风格主文件，新版方法正文位于 `sections/method.tex`。方法章节按照“质量可靠观测 → 图内反事实干预 → 严格共享验证 → 双假设证据账本 → 冲突引导细化 → 因果损失”的顺序展开，保守统计约 2680 个英文词，并包含 17 个方法公式；连同引言中的问题定义，编译后的正文保留 18 个核心公式。

`main.tex` 中由 `\iffalse ... \fi` 包围的内容是旧版方法草稿，仅用于修订追踪，不参与 LaTeX 编译。正式投稿前可以在版本管理确认新版稳定后删除该段。

当前稿件是“可执行的投稿前草稿”，不是可以直接宣称 SOTA 的最终论文。所有未实测数字均为红色 `TBD`。任何没有真实日志支持的结果都不应写入论文。

## 编译

```bash
latexmk -pdf main.tex
```

若只安装了基本 TeX：

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 实验完成后的回填

```bash
python ../tools/export_latex.py \
  --results ../runs/retrace_default/test.json \
  --output results_auto.tex --prefix Main
```

跨数据集、鲁棒性和消融表需要分别运行对应配置后填写。主结果应使用固定阈值或只在验证集选择的阈值；测试集 oracle 阈值只能作为诊断结果。

新版方法除常规 F1/IoU 外，还必须报告 manipulated causal drop、background invariance、intervention precision、ledger contradiction rate 和 uncertainty-error AUROC。没有这些机制指标，反事实与证据账本只能算结构描述，无法形成完整论证。

投稿前检查：

```bash
rg "TBD|Results pending|Anonymous" .
```

还必须替换作者信息、加入真实定量结果和可视化、补充三次随机种子统计、核验参考文献页码/DOI，并根据最终实验删改所有未经证据支持的结论。
