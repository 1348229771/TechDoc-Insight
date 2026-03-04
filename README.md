# 英文技术文档摘要生成系统（T5 微调与多端部署）

面向计算机科学领域的英文技术文档自动摘要系统，基于 `t5-base` 架构进行深度微调。本项目提供从数据处理、模型训练到多端部署（桌面 GUI + Web UI）的完整解决方案。

核心亮点：
- **双模对比**：一键对比原版 T5 与微调后模型的摘要质量，提供 ROUGE/BLEU 指标、可视化图表及详细的差异报告。
- **多端交互**：
  - **Web 端**：基于 Streamlit 的现代化在线界面，支持参数实时调节、文件上传及双向翻译。
  - **桌面端**：基于 Tkinter 的四栏布局 GUI，专为人工质检和对照阅读设计。
- **本地化适配**：针对 Windows 环境优化，支持模型/分词器的一键离线缓存与镜像加速，解决中文路径和网络连接问题。
- **全流程保障**：包含完整的训练/评估流水线，以及覆盖单元测试与集成测试的自动化测试框架。

## 目标与亮点
- **深度微调**：在特定领域数据上微调 T5，显著提升专业文档的摘要准确性。
- **质量评估**：自动生成包含 ROUGE-1/2/L、BLEU 分数的详细评估报告。
- **便捷部署**：一键脚本下载所需模型资源，自动处理环境依赖。
- **双向翻译**：集成 Helsinki-NLP 翻译模型，支持输入/输出的英↔中互译，辅助跨语言理解。
- **工程规范**：提供完整的测试用例、测试计划与 BUG 报告模板，确保系统稳定性。

## 项目结构
```
.
├─ configs/                               # 全局配置
│  └─ config.py                           # 模型名、缓存目录、训练/推理参数、数据路径
├─ docs/                                  # 文档
│  └─ testing/                            # 测试文档
│     ├─ 测试计划.md                       # 测试策略与范围
│     ├─ 测试用例.md                       # 详细测试步骤
│     ├─ 测试执行报告.md                   # 测试结果与指标
│     └─ BUG报告.md                        # 缺陷报告模板
├─ models/                                # 模型与推理
│  ├─ cache/                              # 本地缓存的模型（中文路径时自动回退到 C:\hf_cache）
│  ├─ inference.py                        # T5Summarizer 推理封装（预处理/生成/后处理/批量）
│  └─ t5_model.py                         # 加载/训练/保存（fp16、梯度裁剪、学习率调度、最佳模型）
├─ scripts/                               # 脚本入口
│  ├─ download_assets.py                  # 一键下载并校验 T5 与翻译模型（镜像与离线缓存统一）
│  ├─ gui_summarizer.py                   # 桌面 GUI（Tkinter 四栏：输入/摘要/原文翻译/摘要翻译）
│  ├─ web_ui.py                           # Web UI（Streamlit 在线版：参数调节/文件上传/双向翻译）
│  ├─ train.py                            # 训练/评估主入口（显存自适应、历史保存）
│  └─ compare_t5.py                       # 原版 vs 微调 对比评估 + 可视化 + 报告/CSV 导出
├─ utils/                                 # 工具模块
│  ├─ data_loader.py                      # JSON 数据加载、DataLoader 构造、分词器加载
│  └─ metrics.py                          # ROUGE/BLEU 计算与评估流程
├─ outputs/                               # 训练与评估输出
│  ├─ models/                             # 权重与训练历史
│  │  ├─ best_model/                      # 验证最优模型权重与配置
│  │  └─ final_model/                     # 最终模型权重与配置
│  └─ results/                            # 对比与评估结果
│     ├─ comparison_results.json          # 指标与预测/参考文本
│     ├─ comparison_metrics.png           # 指标柱状图（数值标注）
│     ├─ comparison_report.md             # 人类友好报告（含差值 Δ 与示例）
│     └─ final_evaluation_results.json    # 训练后最终评估结果
├─ tests/                                 # 测试代码
│  ├─ test_data_loader.py                 # 数据加载测试
│  ├─ test_metrics.py                     # 评估指标测试
│  └─ test_pipeline.py                    # 完整流程测试
├─ README.md                              # 项目说明
├─ requirements.txt                       # 依赖清单
└─ run_tests.py                           # 测试运行入口
```

目录说明：
- configs：全局配置（模型名、缓存、训练/推理参数、数据路径）
- docs：项目文档，包含详细的测试文档
- models：摘要模型与推理封装，包含本地缓存目录
- scripts：训练、评估、GUI 与资源下载脚本
- utils：数据加载与评估指标
- outputs：训练与评估的输出结果
- tests：单元测试与集成测试代码

## 安装与准备
- 环境：Python ≥ 3.8（建议虚拟环境）
- 依赖安装：
```
pip install -r requirements.txt
```
- 国内镜像：
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- 一键下载模型与分词器：
```
python scripts/download_assets.py
```
说明：
- 统一镜像端点与缓存：`HF_ENDPOINT`、`HF_HOME`
- 中文路径下的 `sentencepiece` 限制已通过 ASCII 路径回退解决（见 `configs/config.py`）
- 翻译模型 safetensors 损坏自动回退到 `pytorch_model.bin`

## 数据集
- 目录：`data/reconstructed_data/`
- 文件：
  - `train_preprocessed.json`
  - `val_preprocessed.json`
  - `test_preprocessed.json`
- JSON 结构示例：
```
[
  {"input": "<原文文本>", "target": "<摘要文本>"},
  ...
]
```

## 训练与评估
- 训练（显存自适应、CUDA 优先）：
```
python scripts/train.py --use_gpu
```
- 仅评估：
```
python scripts/train.py --evaluate_only --use_gpu
```
- 常用参数：
```
--model_name t5-base
--output_dir outputs
--num_epochs 3
--batch_size 4
--learning_rate 1e-4
--resume_from_checkpoint outputs/models/best_model
--fast_dev_run / --short_run
```
训练输出：
- `outputs/models/best_model/`：最优权重
- `outputs/models/final_model/`：最终权重
- `outputs/logs/train_history.json`：训练曲线
- `outputs/results/final_evaluation_results.json`：最终评估

## 自动化测试
项目包含完整的单元测试与集成测试，确保代码质量与系统稳定性。
- 运行所有测试：
```
python run_tests.py
```
- 或使用 unittest：
```
python -m unittest discover tests
```
- 测试覆盖范围：
  - 数据加载与预处理 (`test_data_loader.py`)
  - 评估指标计算 (`test_metrics.py`)
  - 完整推理流水线 (`test_pipeline.py`)
- 详细文档见 `docs/testing/` 目录。

## 原版 vs 微调 对比实验（核心）
一键对比原版与微调的摘要质量，并输出多种结果格式：
```
python scripts/compare_t5.py --use_gpu --fp16 \
  --plot_output outputs/results/comparison_metrics.png \
  --report_output outputs/results/comparison_report.md \
  --csv_output outputs/results/comparison_results.csv
```
输出内容：
- `outputs/results/comparison_results.json`：完整指标与预测/参考文本
- `outputs/results/comparison_metrics.png`：柱状图（顶部显示具体数值）
- `outputs/results/comparison_report.md`：人类友好报告（包含差值 Δ 与示例）
- `outputs/results/comparison_results.csv`：结构化表（易导入 Excel/R）
控制台输出：
- 指标对比表（Pretrained / Fine-tuned / Δ）
- 示例摘要预览（可通过 `--examples` 控制数量）
可配置项：
- `--metrics rouge1 rouge2 rougeL bleu` 指定度量集合
- `--resume_from_checkpoint outputs/models/best_model` 指定微调权重
- `--max_test_samples` 控制评估样本数

## 桌面 GUI（四栏翻译视图）
```
python scripts/gui_summarizer.py
```
- 左上：输入英文技术文档
- 右上：生成摘要
- 左下：原文翻译（英→中或中→英）
- 右下：摘要翻译（与左下同步方向）
- 首次打开即完整可见；四框等分；底部按钮固定且不遮挡

## Web 端在线摘要生成器
基于 Streamlit 开发的现代化 Web 界面，提供更丰富的功能和交互体验。
- **启动方式**：
  ```bash
  streamlit run scripts/web_ui.py
  # 或者在 Windows 下直接运行批处理文件
  start_web_ui.bat
  ```
- **核心功能**：
  - **模型配置**：支持在界面侧边栏切换“原版 (Pretrained)”和“微调 (Fine-tuned)”模型，并实时调整 `max_length` 和 `num_beams` 参数。
  - **多模态输入**：支持直接粘贴文本或上传 `.txt` 文件进行处理。
  - **双向翻译**：集成 Helsinki-NLP 翻译模型，支持一键翻译“原文”和“生成的摘要”。
  - **结果展示**：分栏展示输入、摘要、原文翻译和摘要翻译，方便对照阅读。

## 配置说明（configs/config.py）
- `MODEL_NAME`：默认 `t5-base`
- `MODEL_CACHE_DIR`：中文路径自动回退到 `C:\hf_cache`
- `TRAIN_CONFIG`、`DATA_CONFIG`、`INFERENCE_CONFIG`：统一管理参数

## 常见问题
- `sentencepiece` 在中文路径报错
  - 已自动回退到 ASCII 缓存路径；如需自定义，请修改 `MODEL_CACHE_DIR`
- safetensors 头损坏
  - 脚本与 GUI 已自动回退至 `pytorch_model.bin`
- PowerShell 中 `conda` 未识别
  - 执行 `D:/rj/anaconda3/Scripts/conda.exe init powershell` 并重启

## 许可证
学术与教学用途优先；商业使用请遵循第三方依赖许可（Transformers、PyTorch 等）。

