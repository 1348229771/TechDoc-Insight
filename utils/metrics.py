"""
评估指标模块
实现ROUGE、BLEU等摘要评估指标
"""

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import torch
from transformers import T5Tokenizer


class SummarizationMetrics:
    """摘要评估指标类"""
    
    def __init__(self, rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=True):
        """
        初始化评估指标
        
        Args:
            rouge_types: ROUGE指标类型列表
            use_stemmer: 是否使用词干提取
        """
        self.rouge_types = rouge_types
        self.rouge_scorer = rouge_scorer.RougeScorer(
            rouge_types, use_stemmer=use_stemmer
        )
    
    def compute_rouge(self, predictions, references):
        """
        计算ROUGE分数
        
        Args:
            predictions: 预测摘要列表
            references: 参考摘要列表
            
        Returns:
            ROUGE分数字典
        """
        # 初始化结果字典
        rouge_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        # 计算每个样本的ROUGE分数
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        # 计算平均分数
        avg_rouge_scores = {}
        for rouge_type in self.rouge_types:
            avg_rouge_scores[rouge_type] = np.mean(rouge_scores[rouge_type])
        
        return avg_rouge_scores
    
    def compute_bleu(self, predictions, references, smooth_method="exp", smooth_value=None):
        """
        计算BLEU分数
        
        Args:
            predictions: 预测摘要列表
            references: 参考摘要列表
            smooth: 是否使用平滑
            
        Returns:
            BLEU分数
        """
        # sacrebleu期望参考摘要是一个列表的列表（每个预测可以有多个参考）
        # 这里我们每个预测只有一个参考摘要
        bleu_score = corpus_bleu(
            predictions,
            [references],
            smooth_method=smooth_method,
            smooth_value=smooth_value
        ).score
        return bleu_score
    
    def compute_all_metrics(self, predictions, references):
        """
        计算所有评估指标
        
        Args:
            predictions: 预测摘要列表
            references: 参考摘要列表
            
        Returns:
            所有评估指标的字典
        """
        rouge_scores = self.compute_rouge(predictions, references)
        bleu_score = self.compute_bleu(predictions, references)
        
        # 合并所有指标
        all_metrics = rouge_scores.copy()
        all_metrics["bleu"] = bleu_score
        
        return all_metrics


def evaluate_model(model, tokenizer, data_loader, device, max_length=150, num_beams=4):
    """
    评估模型在数据集上的性能
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        data_loader: 数据加载器
        device: 设备
        max_length: 生成摘要的最大长度
        num_beams: 束搜索大小
        
    Returns:
        评估结果
    """
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 生成摘要
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            # 解码预测和参考摘要
            batch_predictions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # 获取参考摘要（需要将-100替换为pad_token_id）
            labels = batch['labels'].clone()
            labels[labels == -100] = tokenizer.pad_token_id
            batch_references = tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )
            
            # 添加到结果列表
            predictions.extend(batch_predictions)
            references.extend(batch_references)
    
    # 计算评估指标
    metrics_calculator = SummarizationMetrics()
    metrics = metrics_calculator.compute_all_metrics(predictions, references)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references
    }


def print_evaluation_results(results):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
    """
    metrics = results['metrics']
    
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    
    # 打印ROUGE分数
    for rouge_type in ["rouge1", "rouge2", "rougeL"]:
        if rouge_type in metrics:
            print(f"{rouge_type.upper()}: {metrics[rouge_type]:.4f}")
    
    # 打印BLEU分数
    if "bleu" in metrics:
        print(f"BLEU: {metrics['bleu']:.4f}")
    
    print("="*50)
    
    # 打印一些示例
    print("\n示例摘要:")
    print("-"*50)
    
    predictions = results['predictions'][:3]
    references = results['references'][:3]
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        print(f"\n示例 {i+1}:")
        print(f"参考摘要: {ref}")
        print(f"预测摘要: {pred}")


def save_evaluation_results(results, output_path):
    """
    保存评估结果到文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    import json
    
    # 准备保存的数据
    save_data = {
        'metrics': results['metrics'],
        'predictions': results['predictions'],
        'references': results['references']
    }
    
    # 保存到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    from transformers import T5ForConditionalGeneration
    from configs.config import TEST_DATA_PATH, MODEL_NAME, MODEL_CACHE_DIR, INFERENCE_CONFIG
    
    # 加载模型和分词器
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 创建测试数据加载器
    from utils.data_loader import create_data_loaders
    _, _, test_loader = create_data_loaders(
        None, None, TEST_DATA_PATH, tokenizer, batch_size=4
    )
    
    # 评估模型
    results = evaluate_model(
        model, tokenizer, test_loader, device,
        max_length=INFERENCE_CONFIG["max_length"],
        num_beams=INFERENCE_CONFIG["num_beams"]
    )
    
    # 打印结果
    print_evaluation_results(results)
