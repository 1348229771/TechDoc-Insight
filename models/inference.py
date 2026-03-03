"""
模型推理模块
实现T5模型的推理功能，包括文本预处理、摘要生成和后处理
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import MODEL_NAME, MODEL_CACHE_DIR, INFERENCE_CONFIG
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = MODEL_CACHE_DIR

# 设置本地模型路径
LOCAL_MODEL_PATH = "D:\\t5-model"

import torch
import numpy as np
from transformers import AutoTokenizer
from models.t5_model import T5SummarizationModel
import re


class T5Summarizer:
    """T5摘要生成器"""
    
    def __init__(self, model_path=None, tokenizer_name=None, device=None):
        """
        初始化T5摘要生成器
        
        Args:
            model_path: 模型路径，如果为None则使用预训练模型
            tokenizer_name: 分词器名称，如果为None则使用MODEL_NAME
            device: 计算设备
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_name = tokenizer_name if tokenizer_name else MODEL_NAME
        
        local_tokenizer_path = None
        cache_dir = MODEL_CACHE_DIR
        candidate = os.path.join(cache_dir, self.tokenizer_name)
        if os.path.exists(candidate):
            local_tokenizer_path = candidate
        if local_tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, use_fast=True, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, cache_dir=MODEL_CACHE_DIR)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            self.model = T5SummarizationModel(
                model_name=self.tokenizer_name,
                cache_dir=MODEL_CACHE_DIR,
                device=self.device
            )
            self.model.load_model(model_path)
            print(f"已从{model_path}加载模型")
        else:
            self.model = T5SummarizationModel(
                model_name=self.tokenizer_name,
                cache_dir=MODEL_CACHE_DIR,
                device=self.device
            )
            print("已加载预训练模型")
    
    def preprocess_text(self, text, max_source_length=512):
        """
        预处理文本
        
        Args:
            text: 原始文本
            max_source_length: 最大源文本长度
            
        Returns:
            处理后的输入ID和注意力掩码
        """
        # 添加T5任务前缀
        if not text.startswith("summarize: "):
            text = "summarize: " + text
        
        # 分词
        inputs = self.tokenizer(
            text,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 移到设备
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        return input_ids, attention_mask
    
    def postprocess_text(self, generated_ids):
        """
        后处理生成的文本
        
        Args:
            generated_ids: 生成的ID
            
        Returns:
            处理后的文本
        """
        # 解码
        summary = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 移除多余的空格
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # 确保句子以大写字母开头
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # 确保句子以句号结尾
        if summary and not summary.endswith('.'):
            summary += '.'
        
        return summary
    
    def generate_summary(self, text, max_length=150, min_length=50, num_beams=4, 
                        early_stopping=True, no_repeat_ngram_size=2, length_penalty=2.0,
                        max_source_length=512):
        """
        生成摘要
        
        Args:
            text: 原始文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            num_beams: 束搜索大小
            early_stopping: 是否提前停止
            no_repeat_ngram_size: 不重复的n-gram大小
            length_penalty: 长度惩罚
            max_source_length: 最大源文本长度
            
        Returns:
            生成的摘要
        """
        # 预处理
        input_ids, attention_mask = self.preprocess_text(text, max_source_length)
        
        # 生成摘要
        generated_ids = self.model.generate_summary(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty
        )
        
        # 后处理
        summary = self.postprocess_text(generated_ids[0])
        
        return summary
    
    def generate_summaries_batch(self, texts, batch_size=8, **kwargs):
        """
        批量生成摘要
        
        Args:
            texts: 文本列表
            batch_size: 批量大小
            **kwargs: 生成摘要的其他参数
            
        Returns:
            摘要列表
        """
        summaries = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 预处理
            batch_inputs = []
            batch_masks = []
            
            for text in batch_texts:
                input_ids, attention_mask = self.preprocess_text(text, kwargs.get("max_source_length", 512))
                batch_inputs.append(input_ids)
                batch_masks.append(attention_mask)
            
            # 合并为批次
            batch_input_ids = torch.cat(batch_inputs, dim=0)
            batch_attention_mask = torch.cat(batch_masks, dim=0)
            
            # 生成摘要
            batch_generated_ids = self.model.generate_summary(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                **kwargs
            )
            
            # 后处理
            for generated_ids in batch_generated_ids:
                summary = self.postprocess_text(generated_ids)
                summaries.append(summary)
        
        return summaries
    
    def summarize_from_file(self, file_path, output_path=None, **kwargs):
        """
        从文件读取文本并生成摘要
        
        Args:
            file_path: 输入文件路径
            output_path: 输出文件路径，如果为None则不保存
            **kwargs: 生成摘要的其他参数
            
        Returns:
            生成的摘要
        """
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 生成摘要
        summary = self.generate_summary(text, **kwargs)
        
        # 保存摘要
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"摘要已保存到: {output_path}")
        
        return summary
    
    def evaluate_summary_quality(self, text, reference_summary, **kwargs):
        """
        评估摘要质量
        
        Args:
            text: 原始文本
            reference_summary: 参考摘要
            **kwargs: 生成摘要的其他参数
            
        Returns:
            生成的摘要和评估指标
        """
        # 生成摘要
        generated_summary = self.generate_summary(text, **kwargs)
        
        # 计算ROUGE分数
        from utils.metrics import SummarizationMetrics
        metrics = SummarizationMetrics()
        rouge_scores = metrics.compute_rouge(generated_summary, reference_summary)
        
        return {
            "generated_summary": generated_summary,
            "reference_summary": reference_summary,
            "rouge_scores": rouge_scores
        }


# 示例使用
if __name__ == "__main__":
    # 创建摘要生成器
    summarizer = T5Summarizer()
    
    # 示例文本
    example_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
    concerned with the interactions between computers and human language, in particular how to program computers
    to process and analyze large amounts of natural language data. The goal is a computer capable of
    "understanding" the contents of documents, including the contextual nuances of the language within them.
    The technology can then accurately extract information and insights contained in the documents as well as
    categorize and organize the documents themselves. Challenges in natural language processing frequently
    involve speech recognition, natural language understanding, and natural-language generation.
    """
    
    # 生成摘要
    summary = summarizer.generate_summary(example_text)
    print("原始文本:", example_text)
    print("生成的摘要:", summary)
    
    # 批量生成摘要
    texts = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
    ]
    summaries = summarizer.generate_summaries_batch(texts)
    print("批量生成的摘要:", summaries)
