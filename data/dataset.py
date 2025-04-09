import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision.transforms.functional import to_tensor
import torch.multiprocessing as mp
from torchvision.io import read_image

class VLMDataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_file,
        transform=None,
        max_text_length=128,
        cache_size=1000
    ):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录路径
            annotation_file: 标注文件路径（JSON格式）
            transform: 图像转换
            max_text_length: 文本最大长度
            cache_size: 图像缓存大小
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_text_length = max_text_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        # COCO数据集格式处理
        self.annotations = []
        self.image_id_to_idx = {}
        
        # 创建图像ID到文件名的映射
        image_id_to_filename = {
            img['id']: img['file_name']
            for img in annotations['images']
        }
        
        # 处理标注
        for idx, ann in enumerate(annotations['annotations']):
            image_id = ann['image_id']
            if image_id in image_id_to_filename:
                self.annotations.append({
                    'image_id': image_id,
                    'image_filename': image_id_to_filename[image_id],
                    'caption': ann['caption']
                })
                self.image_id_to_idx[image_id] = idx
        
        # 初始化图像缓存
        self.cache_size = cache_size
        self.image_cache = {}
        self.cache_lock = mp.Lock()
        
    def __len__(self):
        return len(self.annotations)
    
    def _load_image(self, image_path):
        """加载图像并且进行transform"""
        try:
            image = read_image(image_path)
            image = image.float() / 255.0
            if self.transform is not None:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation["image_id"]
        
        # 检查缓存
        with self.cache_lock:
            if image_id in self.image_cache:
                image = self.image_cache[image_id]
            else:
                image_path = os.path.join(self.image_dir, annotation["image_filename"])
                image = self._load_image(image_path)
                if image is not None:
                    # 更新缓存
                    if len(self.image_cache) >= self.cache_size:
                        # 移除最旧的缓存项
                        oldest_key = next(iter(self.image_cache))
                        del self.image_cache[oldest_key]
                    self.image_cache[image_id] = image
        
        if image is None:
            # 如果图像加载失败，返回一个占位符
            image = torch.zeros(3, 224, 224)
            
        return {
            "image": image,
            "text": annotation["caption"],
            "image_id": image_id
        }
        
    def get_image_by_id(self, image_id):
        """通过图像ID获取数据样本"""
        if image_id not in self.image_id_to_idx:
            raise ValueError(f"Image ID {image_id} not found in dataset")
            
        idx = self.image_id_to_idx[image_id]
        return self[idx] 