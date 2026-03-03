"""
T5模型实现
包含T5模型的加载、微调和保存功能
"""

import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import json

# 设置环境变量以使用国内镜像源
from configs.config import MODEL_CACHE_DIR
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = MODEL_CACHE_DIR

# 设置本地模型路径
LOCAL_MODEL_PATH = "D:\\t5-model"


class T5SummarizationModel:
    """T5摘要模型类"""
    
    def __init__(self, model_name="t5-base", cache_dir=None, device=None, use_fp16=False):
        """
        初始化T5模型
        
        Args:
            model_name: 预训练模型名称
            cache_dir: 模型缓存目录
            device: 计算设备
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        
        # 加载模型和配置
        self.model = None
        self.config = None
        self._load_model()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_fp16)
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
    
    def _load_model(self):
        """加载预训练模型"""
        model_path = None
        if self.cache_dir:
            candidate = os.path.join(self.cache_dir, self.model_name)
            if os.path.exists(candidate):
                model_path = candidate
        if model_path is None:
            model_path = self.model_name
        use_local_only = os.path.isdir(model_path)
        self.config = T5Config.from_pretrained(model_path, cache_dir=self.cache_dir, local_files_only=use_local_only)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, cache_dir=self.cache_dir, local_files_only=use_local_only)
        
        # 将模型移到指定设备
        self.model.to(self.device)
        
        print(f"模型已加载到设备: {self.device}")
        print(f"模型参数量: {self.model.num_parameters():,}")
    
    def setup_optimizer(self, learning_rate=3e-4, weight_decay=0.01):
        """
        设置优化器
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
            
        Returns:
            优化器
        """
        # 设置优化器，不衰减bias和LayerNorm权重
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        return optimizer
    
    def setup_scheduler(self, optimizer, num_training_steps, warmup_steps=0):
        """
        设置学习率调度器
        
        Args:
            optimizer: 优化器
            num_training_steps: 总训练步数
            warmup_steps: 预热步数
            
        Returns:
            学习率调度器
        """
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
        return scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler=None, gradient_accumulation_steps=1, max_grad_norm=1.0):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        valid_steps = 0
        nan_batches = 0
        progress_bar = tqdm(train_loader, desc="训练")
        
        for step, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            if not torch.isfinite(loss):
                nan_batches += 1
                progress_bar.set_postfix({"loss": "nan"})
                for group in optimizer.param_groups:
                    group['lr'] = max(group['lr'] * 0.5, 1e-6)
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # 梯度累积
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            # 反向传播
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新参数
            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.use_fp16:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            valid_steps += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / max(valid_steps, 1)
        return avg_loss
    
    def evaluate(self, eval_loader):
        """
        评估模型
        
        Args:
            eval_loader: 评估数据加载器
            
        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0.0
        valid_steps = 0
        progress_bar = tqdm(eval_loader, desc="评估")
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                if self.use_fp16:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                loss = outputs.loss
                if not torch.isfinite(loss):
                    progress_bar.set_postfix({"loss": "nan"})
                    continue
                total_loss += loss.item()
                valid_steps += 1
                progress_bar.set_postfix({"loss": loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / max(valid_steps, 1)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=3e-4, 
              weight_decay=0.01, warmup_steps=0, gradient_accumulation_steps=1, 
              max_grad_norm=1.0, save_steps=None, output_dir=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
            save_steps: 保存步数
            output_dir: 输出目录
            
        Returns:
            训练历史
        """
        # 设置优化器和调度器
        optimizer = self.setup_optimizer(learning_rate, weight_decay)
        
        # 计算总训练步数
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = self.setup_scheduler(optimizer, total_steps, warmup_steps)
        
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 训练历史
        train_history = {
            "train_loss": [],
            "val_loss": []
        }
        
        # 最佳模型
        best_val_loss = float('inf')
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(
                train_loader, optimizer, scheduler, gradient_accumulation_steps, max_grad_norm
            )
            
            # 评估
            val_loss = self.evaluate(val_loader)
            
            # 记录历史
            train_history["train_loss"].append(train_loss)
            train_history["val_loss"].append(val_loss)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss and output_dir:
                best_val_loss = val_loss
                self.save_model(os.path.join(output_dir, "best_model"))
                print("已保存最佳模型")
            
            # 定期保存模型
            if save_steps and (epoch + 1) % save_steps == 0 and output_dir:
                self.save_model(os.path.join(output_dir, f"model_epoch_{epoch + 1}"))
                print(f"已保存第{epoch + 1}轮模型")
        
        # 保存训练历史
        if output_dir:
            with open(os.path.join(output_dir, "train_history.json"), "w") as f:
                json.dump(train_history, f, indent=2)
        
        return train_history
    
    def save_model(self, output_dir):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型和配置
        self.model.save_pretrained(output_dir, safe_serialization=False)
        self.config.save_pretrained(output_dir)
        
        print(f"模型已保存到: {output_dir}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        # 加载配置
        self.config = T5Config.from_pretrained(model_path)
        
        # 加载模型
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            config=self.config
        )
        
        # 将模型移到设备
        self.model.to(self.device)
        
        print(f"已从{model_path}加载模型")
    
    def generate_summary(self, input_ids, attention_mask, max_length=150, min_length=50, 
                        num_beams=4, early_stopping=True, no_repeat_ngram_size=2, 
                        length_penalty=2.0):
        """
        生成摘要
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            max_length: 最大长度
            min_length: 最小长度
            num_beams: 束搜索大小
            early_stopping: 是否提前停止
            no_repeat_ngram_size: 不重复的n-gram大小
            length_penalty: 长度惩罚
            
        Returns:
            生成的摘要ID
        """
        self.model.eval()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty
            )
        
        return generated_ids


if __name__ == "__main__":
    # 测试代码
    from configs.config import MODEL_NAME, MODEL_CACHE_DIR, TRAIN_CONFIG
    
    # 创建模型
    model = T5SummarizationModel(MODEL_NAME, MODEL_CACHE_DIR)
    
    # 打印模型信息
    print(f"模型名称: {model.model_name}")
    print(f"设备: {model.device}")
    print(f"参数量: {model.model.num_parameters():,}")
