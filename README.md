# 📑 英文技术文档摘要生成系统 (TechDoc Abstract Generation System)

> **深度微调，精准摘要 —— 基于 T5 + Transfer Learning 的计算机科学领域长文本智能摘要平台**

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8%2B-green.svg) ![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)

---

## 🌟 产品简介

**面向计算机科学领域的英文技术文档自动摘要系统**，基于 `t5-base` 架构进行深度微调。本项目提供从数据处理、模型训练到多端部署（桌面 GUI + Web UI）的完整解决方案。

---

## ✨ 核心亮点

### 1. 🔄 双模对比 (Dual-Mode Comparison)
- **一键对比**：支持原版 T5 与微调后模型的摘要质量对比。
- **多维评估**：提供 ROUGE/BLEU 指标、可视化图表及详细的差异报告。

### 2. 🖥️ 多端交互 (Multi-Platform Interaction)
- **Web 端**：基于 **Streamlit** 的现代化在线界面，支持参数实时调节、文件上传及双向翻译。
- **桌面端**：基于 **Tkinter** 的四栏布局 GUI，专为人工质检和对照阅读设计。

### 3. 🌏 本地化适配 (Localization)
- **环境优化**：针对 Windows 环境优化，支持模型/分词器的一键离线缓存与镜像加速。
- **无障碍运行**：解决中文路径和网络连接问题，确保国内环境流畅使用。

### 4. 🛡️ 全流程保障 (Full Process Assurance)
- **自动化测试**：包含完整的训练/评估流水线，以及覆盖单元测试与集成测试的自动化测试框架。
- **双向翻译**：集成 Helsinki-NLP 翻译模型，支持输入/输出的英↔中互译，辅助跨语言理解。

---

## 🚀 快速开始

### 环境要求
- Python 3.8+ (建议虚拟环境)
- PyTorch (建议 GPU 环境)

### 1. 安装依赖
```bash
pip install -r requirements.txt
# 国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 资源准备
一键下载模型与分词器（自动处理镜像与缓存）：
```bash
python scripts/download_assets.py
```

### 3. 启动应用
**方式一：Web 端在线摘要生成器**
```bash
streamlit run scripts/web_ui.py
# Windows 用户也可直接运行: start_web_ui.bat
```

**方式二：桌面 GUI (四栏翻译视图)**
```bash
python scripts/gui_summarizer.py
```

---

## 📊 训练与评估

### 模型训练
支持显存自适应与 CUDA 优先：
```bash
python scripts/train.py --use_gpu --model_name t5-base --num_epochs 3 --batch_size 4
```

### 核心实验：原版 vs 微调对比
一键生成对比报告与可视化图表：
```bash
python scripts/compare_t5.py --use_gpu --fp16 \
  --plot_output outputs/results/comparison_metrics.png \
  --report_output outputs/results/comparison_report.md \
  --csv_output outputs/results/comparison_results.csv
```
> **输出内容**：包含指标对比表、柱状图 (`comparison_metrics.png`)、差异报告 (`comparison_report.md`) 及 CSV 数据。

---

## 📂 项目结构

```
.
├─ configs/           # ⚙️ 全局配置 (模型参数、路径)
├─ docs/              # 📚 项目文档 (测试计划、报告)
├─ models/            # 🧠 模型核心 (T5 封装、推理逻辑、本地缓存)
├─ scripts/           # 🛠️ 执行脚本 (训练、对比、GUI/Web 启动)
├─ utils/             # 🔧 工具模块 (数据加载、Metrics 评估)
├─ outputs/           # 📤 输出产物 (模型权重、日志、评估结果)
├─ tests/             # 🧪 测试代码 (单元测试、集成测试)
├─ requirements.txt   # 📦 依赖清单
└─ run_tests.py       # ✅ 测试入口
```

---

## ⚙️ 配置与数据

### 数据集
位于 `data/reconstructed_data/`，包含训练 (`train`)、验证 (`val`) 和测试 (`test`) 的 JSON 文件。
```json
[
  {"input": "<原文文本>", "target": "<摘要文本>"},
  ...
]
```

### 常用配置 (`configs/config.py`)
- `MODEL_NAME`: 默认 `t5-base`
- `MODEL_CACHE_DIR`: 智能回退机制 (优先本地，中文路径回退至 `C:\hf_cache`)

---

## 🧪 自动化测试

确保系统稳定性，运行所有测试：
```bash
python run_tests.py
```
覆盖范围：数据加载、评估指标、完整推理流水线。

---

## ❓ 常见问题

- **Q: `sentencepiece` 在中文路径报错？**
  - A: 系统已自动回退到 ASCII 缓存路径；无需手动干预。
- **Q: `safetensors` 头损坏？**
  - A: 脚本已自动回退至 `pytorch_model.bin` 加载方式。
- **Q: PowerShell 中 `conda` 未识别？**
  - A: 请执行 `conda init powershell` 并重启终端。

---

## 🤝 许可证

本项目优先用于学术与教学用途。商业使用请遵循第三方依赖许可（Transformers, PyTorch 等）。

*Copyright © 2024 TechDoc Team.*
