"""
训练脚本
用于训练T5摘要模型
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from models.t5_model import T5SummarizationModel
from utils.data_loader import TechnicalDocumentDataset, create_data_loaders, load_tokenizer
from utils.metrics import evaluate_model, print_evaluation_results
from configs.config import (
    MODEL_NAME, MODEL_CACHE_DIR, TRAIN_CONFIG, DATA_CONFIG,
    INFERENCE_CONFIG, EVAL_CONFIG,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SAMPLE_LIMITS
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练T5摘要模型")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, 
                        help="预训练模型名称")
    parser.add_argument("--model_cache_dir", type=str, default=MODEL_CACHE_DIR,
                        help="模型缓存目录")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, default=TRAIN_DATA_PATH,
                        help="训练数据文件路径")
    parser.add_argument("--val_file", type=str, default=VAL_DATA_PATH,
                        help="验证数据文件路径")
    parser.add_argument("--test_file", type=str, default=TEST_DATA_PATH,
                        help="测试数据文件路径")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=TRAIN_CONFIG["num_epochs"],
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"],
                        help="批量大小")
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"],
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"],
                        help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=TRAIN_CONFIG["warmup_steps"],
                        help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=TRAIN_CONFIG["gradient_accumulation_steps"],
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=TRAIN_CONFIG["max_grad_norm"],
                        help="梯度裁剪阈值")
    parser.add_argument("--save_steps", type=int, default=TRAIN_CONFIG["save_steps"],
                        help="保存步数")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="仅进行评估")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                        help="使用GPU")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="混合精度训练")
    parser.add_argument("--max_train_samples", type=int, default=SAMPLE_LIMITS["train"],
                        help="限制训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=SAMPLE_LIMITS["val"],
                        help="限制验证样本数")
    parser.add_argument("--max_test_samples", type=int, default=SAMPLE_LIMITS["test"],
                        help="限制测试样本数")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader工作线程数")
    parser.add_argument("--fast_dev_run", action="store_true", default=False,
                        help="快速调试模式，缩小模型与样本数量以加速")
    parser.add_argument("--short_run", action="store_true", default=False,
                        help="短跑模式，仅限制训练/验证样本数")
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_directory(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子目录
    subdirs = ["models", "logs", "results"]
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    
    return output_dir


def save_config(args, output_dir):
    """保存配置"""
    config_path = os.path.join(output_dir, "config.json")
    
    # 将参数转换为字典
    config = vars(args)
    
    # 添加时间戳
    config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存配置
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"配置已保存到: {config_path}")


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            if props.total_memory <= 8 * 1024 * 1024 * 1024 and args.batch_size > 2:
                args.batch_size = 2
                print("检测到显存较小，调整batch_size到2")
        except Exception:
            pass
    
    # 创建输出目录
    output_dir = create_output_directory(args.output_dir)
    
    # 保存配置
    save_config(args, output_dir)
    
    if args.fast_dev_run:
        if device.type == "cuda":
            args.fp16 = True
        if args.batch_size > 2:
            args.batch_size = 2
        args.num_epochs = 1
        args.max_train_samples = 2000 if args.max_train_samples is None else args.max_train_samples
        args.max_val_samples = 500 if args.max_val_samples is None else args.max_val_samples
        args.max_test_samples = 200 if args.max_test_samples is None else args.max_test_samples
        if args.model_name != "t5-small":
            args.model_name = "t5-small"
        source_len_override = 256
        target_len_override = 128
    elif args.short_run:
        # 仅限制样本数，不改变模型和长度
        args.max_train_samples = 2000 if args.max_train_samples is None else args.max_train_samples
        args.max_val_samples = 500 if args.max_val_samples is None else args.max_val_samples
        args.max_test_samples = 200 if args.max_test_samples is None else args.max_test_samples
        source_len_override = DATA_CONFIG["max_source_length"]
        target_len_override = DATA_CONFIG["max_target_length"]
    else:
        source_len_override = DATA_CONFIG["max_source_length"]
        target_len_override = DATA_CONFIG["max_target_length"]

    tokenizer = load_tokenizer(args.model_name, args.model_cache_dir)
    
    print("加载数据集...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.train_file,
        args.val_file,
        args.test_file,
        tokenizer,
        batch_size=args.batch_size,
        max_source_length=source_len_override,
        max_target_length=target_len_override,
        limit_train_samples=args.max_train_samples,
        limit_val_samples=args.max_val_samples,
        limit_test_samples=args.max_test_samples,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    
    # 创建模型
    print("加载模型...")
    model = T5SummarizationModel(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir,
        device=device,
        use_fp16=args.fp16
    )
    
    # 从检查点恢复训练
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        model.load_model(args.resume_from_checkpoint)
    
    # 仅评估
    if args.evaluate_only:
        print("开始评估...")
        
        # 评估模型
        eval_results = evaluate_model(
            model=model.model,
            tokenizer=tokenizer,
            data_loader=test_loader,
            device=device,
            max_length=INFERENCE_CONFIG["max_length"],
            num_beams=INFERENCE_CONFIG["num_beams"]
        )
        
        # 打印结果
        print_evaluation_results(eval_results)
        
        # 保存结果
        results_path = os.path.join(output_dir, "results", "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"评估结果已保存到: {results_path}")
        return
    
    # 训练模型
    print("开始训练...")
    start_time = time.time()
    
    train_history = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        output_dir=os.path.join(output_dir, "models")
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "models", "final_model")
    model.save_model(final_model_path)
    
    # 评估最终模型
    print("评估最终模型...")
    _, _, test_loader = create_data_loaders(
        args.train_file,
        args.val_file,
        args.test_file,
        tokenizer,
        batch_size=args.batch_size,
        max_source_length=source_len_override,
        max_target_length=target_len_override,
        limit_train_samples=args.max_train_samples,
        limit_val_samples=args.max_val_samples,
        limit_test_samples=args.max_test_samples,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # 评估模型
    eval_results = evaluate_model(
        model=model.model,
        tokenizer=tokenizer,
        data_loader=test_loader,
        device=device,
        max_length=INFERENCE_CONFIG["max_length"],
        num_beams=INFERENCE_CONFIG["num_beams"]
    )
    
    # 打印结果
    print_evaluation_results(eval_results)
    
    # 保存结果
    results_path = os.path.join(output_dir, "results", "final_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"评估结果已保存到: {results_path}")
    
    # 保存训练历史
    history_path = os.path.join(output_dir, "logs", "train_history.json")
    with open(history_path, "w") as f:
        json.dump(train_history, f, indent=2)
    
    print(f"训练历史已保存到: {history_path}")
    
    print("训练流程完成!")


if __name__ == "__main__":
    main()
