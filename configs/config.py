"""
配置文件
包含模型、训练和评估的所有参数配置
"""

import os
import sys

# 项目路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "reconstructed_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 数据文件路径
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_preprocessed.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_preprocessed.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_preprocessed.json")

# 模型配置
MODEL_NAME = "t5-base"
_default_cache = os.path.join(MODEL_DIR, "cache")
try:
    _is_ascii = _default_cache.encode("ascii", "strict")
    MODEL_CACHE_DIR = _default_cache
except Exception:
    MODEL_CACHE_DIR = os.path.join("C:\\", "hf_cache")
if not os.path.isdir(MODEL_CACHE_DIR):
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    except Exception:
        pass
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "t5_finetuned")

# 训练配置
TRAIN_CONFIG = {
    "seed": 42,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
}

# 数据处理配置
DATA_CONFIG = {
    "max_source_length": 512,
    "max_target_length": 150,
    "source_prefix": "summarize: ",
    "padding": "max_length",
    "truncation": True,
    "ignore_pad_token_for_loss": True,
}

# 推理配置
INFERENCE_CONFIG = {
    "max_length": 150,
    "min_length": 50,
    "num_beams": 4,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
    "length_penalty": 2.0,
}

# 评估配置
EVAL_CONFIG = {
    "metrics": ["rouge", "bleu"],
    "rouge_types": ["rouge1", "rouge2", "rougeL"],
    "bleu_smooth": True,
}

# 样本数限制（默认加速训练/评估）
SAMPLE_LIMITS = {
    "train": 2000,
    "val": 500,
    "test": 200,
}

# GUI配置
GUI_CONFIG = {
    "title": "英文技术文档摘要生成系统",
    "window_size": "800x600",
    "example_text": """
    Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. 
    These neural networks are inspired by the structure and function of the human brain, consisting of interconnected nodes or neurons that process information. 
    Deep learning has revolutionized various fields including computer vision, natural language processing, and speech recognition. 
    The key advantage of deep learning is its ability to automatically learn hierarchical representations of data, eliminating the need for manual feature engineering. 
    This approach has led to breakthroughs in image classification, object detection, language translation, and many other tasks. 
    However, deep learning models typically require large amounts of labeled data and significant computational resources for training.
    """
}
