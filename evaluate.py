import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.robust_vlm import RobustVLM
from data.dataset import VLMDataset
from config import Config
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math

class NetworkSimulator:
    """网络模拟器，用于模拟网络带宽限制和丢包"""
    def __init__(self, bandwidth_mbps=10, packet_loss_rate=0.6):
        """
        初始化网络模拟器
        
        Args:
            bandwidth_mbps: 网络带宽，单位Mbps
            packet_loss_rate: 丢包率
        """
        self.bandwidth_mbps = bandwidth_mbps
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth_bytes_per_second = bandwidth_mbps * 1024 * 1024 / 8  # 转换为字节/秒
        
    def simulate_transmission(self, data_size_bytes):
        """
        模拟数据传输
        
        Args:
            data_size_bytes: 数据大小（字节）
        Returns:
            transmission_time: 传输时间（秒）
            is_dropped: 是否被丢弃
        """
        # 模拟丢包
        if np.random.random() < self.packet_loss_rate:
            return 0, True
            
        # 计算传输时间
        transmission_time = data_size_bytes / self.bandwidth_bytes_per_second
        return transmission_time, False

class DirectImageVLM(nn.Module):
    """直接传输图片的VLM模型"""
    def __init__(self, visual_encoder, language_decoder, network_simulator=None):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.language_decoder = language_decoder
        self.network_simulator = network_simulator
        
    def forward(self, image, text, image_id=None, use_cache=True):
        # 直接使用原始图片进行特征提取
        visual_features = self.visual_encoder(image)
        
        # 模拟网络传输
        if self.network_simulator:
            # 计算原始图片大小（假设是RGB图像）
            image_size_bytes = image.numel() * 4  # 4 bytes per float32
            transmission_time, is_dropped = self.network_simulator.simulate_transmission(image_size_bytes)
            if is_dropped:
                # 如果图片被丢弃，使用零向量
                visual_features = torch.zeros_like(visual_features)
            time.sleep(transmission_time)  # 模拟传输延迟
            
        # 语言生成
        output = self.language_decoder(visual_features, text)
        return output

def load_model(checkpoint_path, device, model_type='robust', network_simulator=None):
    """加载训练好的模型"""
    if model_type == 'robust':
        model = RobustVLM(
            visual_encoder=None,
            language_decoder=None,
            visual_dim=Config.VISUAL_DIM,
            compressed_dim=Config.COMPRESSED_DIM,
            dropout_rate=Config.DROPOUT_RATE,
            network_simulator=network_simulator
        )
    else:
        model = DirectImageVLM(
            visual_encoder=None,
            language_decoder=None,
            network_simulator=network_simulator
        )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, dataloader, tokenizer, device, model_type='robust'):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    inference_times = []
    transmission_times = []
    dropped_packets = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type} model"):
            images = batch["image"].to(device)
            texts = batch["text"]
            image_ids = batch["image_id"]
            
            # 文本编码
            text_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # 记录推理时间
            start_time = time.time()
            
            # 前向传播
            outputs = model(
                images,
                text_inputs.input_ids,
                image_ids=image_ids
            )
            
            total_time = time.time() - start_time
            inference_times.append(total_time)
            
            # 计算损失
            loss = criterion(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                text_inputs.input_ids.view(-1)
            )
            total_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(text_inputs.input_ids.cpu().numpy())
    
    # 计算评估指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    avg_inference_time = np.mean(inference_times)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'avg_transmission_time': np.mean(transmission_times) if transmission_times else 0,
        'packet_loss_rate': dropped_packets / len(dataloader) if len(dataloader) > 0 else 0
    }

def plot_comparison(robust_metrics, direct_metrics, save_path):
    """绘制两种模型的对比图表"""
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'avg_inference_time', 'avg_transmission_time']
    robust_values = [robust_metrics[name] for name in metrics_names]
    direct_values = [direct_metrics[name] for name in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.figure(figsize=(15, 6))
    plt.bar(x - width/2, robust_values, width, label='Robust VLM (特征向量)')
    plt.bar(x + width/2, direct_values, width, label='Direct VLM (原始图片)')
    
    plt.xlabel('评估指标')
    plt.ylabel('得分/时间(秒)')
    plt.title('模型性能对比')
    plt.xticks(x, metrics_names, rotation=45)
    plt.ylim(0, max(max(robust_values), max(direct_values)) * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_comparison_report(robust_metrics, direct_metrics, save_path):
    """生成对比评估报告"""
    report = {
        'model_comparison': {
            'robust_vlm': robust_metrics,
            'direct_vlm': direct_metrics,
            'timestamp': str(datetime.now())
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)

def main():
    # 设置设备
    device = torch.device(Config.DEVICE)
    
    # 创建网络模拟器
    network_simulator = NetworkSimulator(
        bandwidth_mbps=10,  # 10 Mbps带宽
        packet_loss_rate=0.6  # 60%丢包率
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 创建数据集
    transform = Config.get_image_transform()
    dataset = VLMDataset(
        image_dir="data/images",
        annotation_file="data/annotations.json",
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # 加载两个模型
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_epoch10.pt")
    robust_model = load_model(checkpoint_path, device, 'robust', network_simulator)
    direct_model = load_model(checkpoint_path, device, 'direct', network_simulator)
    
    # 评估两个模型
    robust_metrics = evaluate_model(robust_model, dataloader, tokenizer, device, 'robust')
    direct_metrics = evaluate_model(direct_model, dataloader, tokenizer, device, 'direct')
    
    # 创建输出目录
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 保存对比结果
    plot_comparison(robust_metrics, direct_metrics, "evaluation_results/comparison.png")
    generate_comparison_report(robust_metrics, direct_metrics, "evaluation_results/comparison_report.json")
    
    # 打印评估结果
    print("\n模型对比结果:")
    print("\nRobust VLM (特征向量传输):")
    print(f"平均推理时间: {robust_metrics['avg_inference_time']:.4f}秒")
    print(f"平均传输时间: {robust_metrics['avg_transmission_time']:.4f}秒")
    print(f"丢包率: {robust_metrics['packet_loss_rate']:.2%}")
    print(f"准确率: {robust_metrics['accuracy']:.4f}")
    print(f"精确率: {robust_metrics['precision']:.4f}")
    print(f"召回率: {robust_metrics['recall']:.4f}")
    print(f"F1分数: {robust_metrics['f1']:.4f}")
    
    print("\nDirect VLM (原始图片传输):")
    print(f"平均推理时间: {direct_metrics['avg_inference_time']:.4f}秒")
    print(f"平均传输时间: {direct_metrics['avg_transmission_time']:.4f}秒")
    print(f"丢包率: {direct_metrics['packet_loss_rate']:.2%}")
    print(f"准确率: {direct_metrics['accuracy']:.4f}")
    print(f"精确率: {direct_metrics['precision']:.4f}")
    print(f"召回率: {direct_metrics['recall']:.4f}")
    print(f"F1分数: {direct_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
