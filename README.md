# VDPA: Variance-driven Dual-Path Attention (Official PyTorch)

> STK (Soft Top-K) · LGA (Learnable Global Attention) · VDFC (Variance-Driven Fusion Controller) · ALP (Adaptive Local Pooling, dynamic class token)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch ≥1.10](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)

本仓库包含 **VDPA 模型最小可复现实现** 与 **统一效率 / 参数量 / FLOPs 基准脚本**。  
This repo provides a minimal VDPA implementation plus unified efficiency/params/FLOPs benchmarks.

---

## 🧭 目录
- [Highlights](#-highlights)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [统一效率基准](#-统一效率基准)
- [依赖](#-依赖)
- [许可](#-许可)
- [引用](#-引用)
- [致谢](#-致谢)

---

## ✨ Highlights
- **双分支注意力**：STK（软化的 Top-K 稀疏局部） × LGA（可学习全局聚合）。
- **样本级融合门控**：VDFC 以 token 方差作为结构离散度信号，自适应调节局部/全局比例。
- **动态类标构造**：ALP 基于邻域余弦权重聚合，生成不引入额外 [CLS] 的 class token。
- **整洁实现**：纯 PyTorch、依赖最小；配套 `timm` 与 VMamba 等基线的统一基准脚本。

---

## 📦 安装

**方式 A｜pip（轻量）**
```bash
pip install -r requirements.txt


---

## 🚀 快速开始
import torch
from vdpapkg.models.vdpa import VDPA

model = VDPA(num_classes=2, depth=10, embed_dim=512, num_heads=8)
x = torch.randn(1, 3, 512, 512)
logits = model(x)
print(logits.shape)  # [1, 2]

## 🧪 统一效率基准

脚本一｜measure_models（导出 CSV，含 timm / VMamba 基线）

python benchmarks/measure_models.py \
  --models vdpa,convnextv2_tiny,maxvit,vmamba_tiny \
  --img 512 --bs 1 --embed 512 --depth 10 --heads 8 \
  --fp16 --threads 8 --csv results_efficiency.csv


脚本二｜bench_models_base（“Base/大杯”体量映射 + VDPA）

python benchmarks/bench_models_base.py \
  --models resnet50,vit,swin-transformer,vdpa \
  --img 512 --bs 1 --fp16

## 🔧 依赖

Python ≥ 3.8

PyTorch ≥ 1.10

（可选）timm、ptflops / thop / fvcore、numpy

## 📄 许可

本项目采用 MIT 许可证（见 LICENSE
）。

## 🔗 引用

如果本项目对你的研究有帮助，请在论文或项目中引用（见 CITATION.cff）。

🙏 致谢

timm、fvcore、ptflops、thop 等优秀开源项目。
