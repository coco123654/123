# 123 — 萌发菌表达量可视化 / Mycena Expression Visualization

本项目基于萌发菌（*Mycena*）与天麻（*Gastrodia elata*）共生 vs 非共生
转录组数据（5–50天时间序列），生成与文献一致的基因表达量折线图。

## 输入数据 Input Files

将以下四张表放在项目根目录，文件名需与下列一致（或参照脚本注释修改）：

| 文件名 | 说明 |
|--------|------|
| `萌发菌与天麻共生5-50天的表达量总表.csv` | 共生条件下各时间点 FPKM/TPM 表达量矩阵 |
| `萌发菌自己非共生5-50天的表达量总表.csv` | 非共生条件下各时间点 FPKM/TPM 表达量矩阵 |
| `萌发菌与天麻共生5-50天的基因计数表达定量注释结果表.csv` | 共生条件基因计数 + 功能注释（含 KEGG/GO/Category 等列） |
| `萌发菌自己非共生5-50天的基因计数表达定量注释结果表.csv` | 非共生条件基因计数 + 功能注释 |

> 若为 Excel 格式（`.xlsx`），脚本同样支持自动识别。

## 安装依赖 Install

```bash
pip install -r requirements.txt
```

## 运行 Usage

```bash
python generate_expression_figure.py
```

## 输出图像 Output Figures

| 文件 | 内容 |
|------|------|
| `expression_figure.png` | 各基因功能类别（MFS转运体、GPCR、氮代谢等）按时间点的平均表达折线图 |
| `key_genes_expression.png` | 关键基因（*MyNrt*、*MyNir*、*MyAmid*、*GPCR_01* 等）个体表达折线图 |
| `category_expression_bar.png` | 各时间点各功能类别平均表达柱状对比图 |

## 参考文献 Reference

Yuan Q-S *et al.* "Fungal symbiont *Mycena* complements impaired nitrogen
utilization in *Gastrodia elata* and supplies indole-3-acetic acid to
facilitate its seed germination." *Plant Communications* **6**(10), 101500 (2025).

