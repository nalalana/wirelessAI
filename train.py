import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.robust_vlm import RobustVLM
from torchvision.models import resnet50
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from data.dataset import VLMDataset
from config import Config
import os
from tqdm import tqdm
from PIL import Image
import time
import psutil
import GPUtil
from torch.cuda.amp import autocast, GradScaler

def print_gpu_usage():
    """打印GPU使用情况"""
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        print(f"  GPU Utilization: {gpu.load*100}%")
        print(f"  Temperature: {gpu.temperature}°C")

def print_memory_usage():
    """打印内存使用情况"""
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def create_model(visual_dim=2048, compressed_dim=512, dropout_rate=0.6):
    # 创建视觉编码器
    visual_encoder = resnet50(weights='IMAGENET1K_V1')  # 更新为新的API
    visual_encoder = nn.Sequential(*list(visual_encoder.children())[:-1])
    
    # 创建语言解码器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # 添加padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    language_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # 创建RobustVLM模型
    model = RobustVLM(
        visual_encoder=visual_encoder,
        language_decoder=language_decoder,
        visual_dim=visual_dim,
        compressed_dim=compressed_dim,
        dropout_rate=dropout_rate
    )
    
    return model, tokenizer

def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    num_epochs=10,
    learning_rate=1e-4,
    save_dir="checkpoints",
    gradient_accumulation_steps=4
):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 初始化梯度缩放器用于混合精度训练
    scaler = GradScaler()
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        # 训练阶段
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 将数据移到GPU
            images = batch["image"].to(device)
            texts = batch["text"]
            image_ids = batch["image_id"]
            
            # 文本编码
            text_inputs = tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=Config.MAX_TEXT_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            # 使用混合精度训练
            with autocast():
                # 前向传播
                outputs = model(
                    images,
                    text_inputs.input_ids,
                    image_id=image_ids,
                    use_cache=True
                )
                
                # 计算损失
                logits = outputs.logits[:, 1:, :]
                targets = text_inputs.input_ids[:, 1:]
                attention_mask = text_inputs.attention_mask[:, 1:]
                
                # 获取非padding token的索引
                non_padding_indices = attention_mask.nonzero(as_tuple=True)
                
                # 只选择非padding token的logits和targets
                logits = logits[non_padding_indices]
                targets = targets[non_padding_indices]
                
                # 计算损失
                loss = criterion(logits, targets)
                loss = loss / gradient_accumulation_steps  # 缩放损失
            
            # 使用梯度缩放器进行反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 更新进度条
            train_steps += 1
            train_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失值
            
            # 验证
            if train_steps % Config.VALIDATION_STEPS == 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_images = val_batch["image"].to(device)
                        val_text_inputs = tokenizer(
                            val_batch["text"],
                            padding="max_length",
                            truncation=True,
                            max_length=Config.MAX_TEXT_LENGTH,
                            return_tensors="pt"
                        ).to(device)
                        val_image_ids = val_batch["image_id"]
                        
                        val_outputs = model(
                            val_images,
                            val_text_inputs.input_ids,
                            image_id=val_image_ids,
                            use_cache=True
                        )
                        
                        # 计算验证损失
                        val_logits = val_outputs.logits[:, 1:, :]
                        val_targets = val_text_inputs.input_ids[:, 1:]
                        val_attention_mask = val_text_inputs.attention_mask[:, 1:]
                        
                        # 获取非padding token的索引
                        val_non_padding_indices = val_attention_mask.nonzero(as_tuple=True)
                        
                        # 只选择非padding token的logits和targets
                        val_logits = val_logits[val_non_padding_indices]
                        val_targets = val_targets[val_non_padding_indices]
                        
                        val_loss += criterion(val_logits, val_targets).item()
                        val_steps += 1
        
        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/train_steps:.4f}")
        print(f"Val Loss: {val_loss/val_steps:.4f}")
        
        # 保存检查点
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss/train_steps,
            "val_loss": val_loss/val_steps
        }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt"))
        
        # # 打印GPU使用情况
        # print("\nEpoch结束时的GPU使用情况:")
        # print_gpu_usage()
        # print_memory_usage()

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model, tokenizer = create_model()
    model = model.to(device)
    
    # 创建数据集
    transform = Config.get_image_transform()
    train_dataset = VLMDataset(
        image_dir=Config.TRAIN_IMAGE_DIR,
        annotation_file=Config.TRAIN_ANNOTATION_FILE,
        transform=transform
    )
    val_dataset = VLMDataset(
        image_dir=Config.VAL_IMAGE_DIR,
        annotation_file=Config.VAL_ANNOTATION_FILE,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # 增加worker数量
        pin_memory=True,  # 使用固定内存
        prefetch_factor=2,  # 预加载因子
        persistent_workers=True,  # 保持worker进程
        drop_last=True  # 丢弃不完整的最后一个batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 开始训练
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device
    )
