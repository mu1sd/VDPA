# VDPA: Variance-driven Dual-Path Attention (Official PyTorch)

> STK (Soft Top-K) + LGA (Learnable Global Attention) + VDFC (Variance-Driven Fusion Controller) + ALP (Adaptive Local Pooling, dynamic class token)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![PyTorch ≥1.10](https://img.shields.io/badge/PyTorch-%E2%89%A51.10-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/downloads/)

本仓库包含 **VDPA 模型最小可复现实现** 与 **统一效率/参数量/FLOPs 基准脚本**。

## ✨ Highlights
- **双分支注意力**：STK（软化的 Top-K 稀疏局部） × LGA（可学习全局池化注意力）。  
- **样本级融合门控**：VDFC 以 token 方差为结构离散度信号，自适应调节局部/全局比例。  
- **动态类标构造**：ALP 基于邻域余弦权重聚合，生成不引入额外 [CLS] 的 class token。  
- **整洁实现**：纯 PyTorch，依赖最小；含 `timm` 与 VMamba 等可对比基线的统一基准脚本。

## 📦 安装
```bash
# 方式A：轻量依赖
pip install -r requirements.txt

# 方式B：Conda（可选）
conda env create -f environment.yml && conda activate vdpa
