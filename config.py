import torch
import torch.nn as nn
from torchvision import transforms
import os

class RobustVLM(nn.Module):
    def __init__(self, visual_encoder, language_decoder):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.language_decoder = language_decoder
        self.dropout = nn.Dropout(p=0.6)  # 模拟60%的丢包率
        
    def forward(self, image, text):
        # 视觉特征提取
        visual_features = self.visual_encoder(image)
        
        # 模拟网络丢包
        if self.training:
            visual_features = self.dropout(visual_features)
            
        # 语言生成
        output = self.language_decoder(visual_features, text)
        return output

class Config:
    # 模型参数
    VISUAL_DIM = 2048
    COMPRESSED_DIM = 512
    DROPOUT_RATE = 0.6
    CACHE_SIZE = 1000
    
    # 训练参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    VALIDATION_STEPS = 2000
    MAX_TEXT_LENGTH = 128
    
    # 数据参数
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    
    # 路径参数
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    DATA_DIR = "data"
    TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "images", "train2017")
    VAL_IMAGE_DIR = os.path.join(DATA_DIR, "images", "val2017")
    TRAIN_ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations/annotations", "captions_train2017.json")
    VAL_ANNOTATION_FILE = os.path.join(DATA_DIR, "annotations/annotations", "captions_val2017.json")
    
    # 设备参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型保存路径
    MODEL_SAVE_PATH = "checkpoints/model.pt"
    
    # 数据加载参数
    NUM_WORKERS = 8
    
    @staticmethod
    def get_image_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
