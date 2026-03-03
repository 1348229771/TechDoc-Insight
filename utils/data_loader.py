"""
数据加载器
负责加载和预处理英文技术文档数据集
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TechnicalDocumentDataset(Dataset):
    """技术文档数据集类"""
    
    def __init__(self, data_path, tokenizer, max_source_length=512, max_target_length=150, 
                 source_prefix="summarize: ", padding="max_length", truncation=True, limit_samples=None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_source_length: 输入文本最大长度
            max_target_length: 目标摘要最大长度
            source_prefix: 输入文本前缀
            padding: 填充策略
            truncation: 是否截断
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = source_prefix
        self.padding = padding
        self.truncation = truncation
        
        # 加载数据
        self.data = self._load_data()
        if isinstance(limit_samples, int) and limit_samples > 0:
            self.data = self.data[:limit_samples]
    
    def _load_data(self):
        """加载JSON格式的数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        item = self.data[idx]
        
        # 获取输入和目标文本
        input_text = item['input']
        target_text = item['target']
        
        # 添加前缀
        if self.source_prefix:
            input_text = self.source_prefix + input_text
        
        # 对输入文本进行编码
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        # 将标签转换为模型输入格式
        # 将填充位置的标签设置为-100，这样在计算损失时会被忽略
        labels["input_ids"] = labels["input_ids"].squeeze()
        model_inputs["labels"] = torch.where(
            labels["input_ids"] == self.tokenizer.pad_token_id,
            torch.tensor(-100),
            labels["input_ids"]
        )
        
        # 移除批处理维度
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor) and model_inputs[key].dim() > 1:
                model_inputs[key] = model_inputs[key].squeeze()
        
        return model_inputs


def create_data_loaders(train_path, val_path, test_path, tokenizer, batch_size=8, 
                        max_source_length=512, max_target_length=150,
                        limit_train_samples=None, limit_val_samples=None, limit_test_samples=None,
                        num_workers=0, pin_memory=True):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        test_path: 测试数据路径
        tokenizer: 分词器
        batch_size: 批大小
        max_source_length: 输入文本最大长度
        max_target_length: 目标摘要最大长度
        
    Returns:
        训练、验证和测试数据加载器
    """
    # 创建数据集
    train_dataset = TechnicalDocumentDataset(
        train_path, tokenizer, max_source_length, max_target_length, limit_samples=limit_train_samples
    )
    val_dataset = TechnicalDocumentDataset(
        val_path, tokenizer, max_source_length, max_target_length, limit_samples=limit_val_samples
    )
    test_dataset = TechnicalDocumentDataset(
        test_path, tokenizer, max_source_length, max_target_length, limit_samples=limit_test_samples
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def load_tokenizer(model_name="t5-base", cache_dir=None):
    """
    加载T5分词器
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
        
    Returns:
        T5分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(cache_dir, model_name) if cache_dir and os.path.exists(os.path.join(cache_dir, model_name)) else model_name,
        cache_dir=cache_dir,
        use_fast=True
    )
    return tokenizer


def get_data_statistics(data_path):
    """
    获取数据集统计信息
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        数据集统计信息
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算输入和目标文本长度
    input_lengths = [len(item['input'].split()) for item in data]
    target_lengths = [len(item['target'].split()) for item in data]
    
    # 计算统计信息
    stats = {
        'num_samples': len(data),
        'input_length': {
            'min': min(input_lengths),
            'max': max(input_lengths),
            'mean': sum(input_lengths) / len(input_lengths),
            'median': sorted(input_lengths)[len(input_lengths) // 2]
        },
        'target_length': {
            'min': min(target_lengths),
            'max': max(target_lengths),
            'mean': sum(target_lengths) / len(target_lengths),
            'median': sorted(target_lengths)[len(target_lengths) // 2]
        }
    }
    
    return stats


if __name__ == "__main__":
    # 测试代码
    from configs.config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, MODEL_NAME, MODEL_CACHE_DIR
    
    # 加载分词器
    tokenizer = load_tokenizer(MODEL_NAME, MODEL_CACHE_DIR)
    
    # 获取数据统计信息
    train_stats = get_data_statistics(TRAIN_DATA_PATH)
    val_stats = get_data_statistics(VAL_DATA_PATH)
    test_stats = get_data_statistics(TEST_DATA_PATH)
    
    print("训练集统计信息:", train_stats)
    print("验证集统计信息:", val_stats)
    print("测试集统计信息:", test_stats)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, tokenizer
    )
    
    # 测试数据加载
    print("\n测试数据加载...")
    for batch in train_loader:
        print("输入ID形状:", batch['input_ids'].shape)
        print("注意力掩码形状:", batch['attention_mask'].shape)
        print("标签形状:", batch['labels'].shape)
        break
