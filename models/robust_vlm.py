import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCompressor(nn.Module):
    def __init__(self, input_dim, compressed_dim):
        super().__init__()
        self.compression = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
            nn.ReLU(),
            nn.LayerNorm(compressed_dim)
        )
        
    def forward(self, x):
        return self.compression(x)

class FeatureDecompressor(nn.Module):
    def __init__(self, compressed_dim, output_dim):
        super().__init__()
        self.decompression = nn.Sequential(
            nn.Linear(compressed_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.decompression(x)

class RobustVLM(nn.Module):
    def __init__(
        self, 
        visual_encoder, 
        language_decoder,
        visual_dim=2048,
        compressed_dim=512,
        dropout_rate=0.6,
        cache_size=100
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.language_decoder = language_decoder
        
        # 特征压缩和解压缩
        self.feature_compressor = FeatureCompressor(visual_dim, compressed_dim)
        self.feature_decompressor = FeatureDecompressor(compressed_dim, visual_dim)
        
        # 视觉到文本的投影
        self.visual_to_text_projection = nn.Linear(
            visual_dim,
            language_decoder.config.hidden_size
        )
        
        # 丢包模拟
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 特征缓存
        self.cache_size = cache_size
        self.feature_cache = {}
        
    def _update_cache(self, image_id, features):
        """更新特征缓存"""
        if len(self.feature_cache) >= self.cache_size:
            # 移除最旧的缓存
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        # 使用detach()和to()的组合来避免不必要的内存复制
        self.feature_cache[image_id] = features.detach()
        
    def _get_cached_features(self, image_id):
        """获取缓存的特征"""
        if image_id in self.feature_cache:
            return self.feature_cache[image_id].to(self.visual_encoder.device)
        return None
        
    def forward(self, image, text, image_id=None, use_cache=True):
        # 视觉特征提取
        if use_cache and image_id is not None:
            cached_features = self._get_cached_features(image_id)
            if cached_features is not None:
                visual_features = cached_features
            else:
                with torch.no_grad():  # 视觉编码器不需要梯度
                    visual_features = self.visual_encoder(image)
                # 调整特征维度
                batch_size = visual_features.size(0)
                visual_features = visual_features.view(batch_size, -1)
                self._update_cache(image_id, visual_features)
        else:
            with torch.no_grad():  # 视觉编码器不需要梯度
                visual_features = self.visual_encoder(image)
            # 调整特征维度
            batch_size = visual_features.size(0)
            visual_features = visual_features.view(batch_size, -1)
        
        # 特征压缩
        compressed_features = self.feature_compressor(visual_features)
        
        # 模拟网络丢包
        if self.training:
            compressed_features = self.dropout(compressed_features)
            
        # 特征解压缩
        decompressed_features = self.feature_decompressor(compressed_features)
        
        # 投影到文本空间
        visual_embeddings = self.visual_to_text_projection(decompressed_features)
        
        # 获取文本嵌入
        text_embeddings = self.language_decoder.transformer.wte(text)
        
        # 合并视觉和文本特征
        combined_embeddings = torch.cat([visual_embeddings.unsqueeze(1), text_embeddings], dim=1)
        
        # 4. 生成位置编码
        position_ids = torch.arange(0, combined_embeddings.size(1), dtype=torch.long, device=combined_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 5. 通过GPT-2生成输出
        output = self.language_decoder(
            inputs_embeds=combined_embeddings,
            position_ids=position_ids,
            return_dict=True
        )
        
        return output
    
    def clear_cache(self):
        """清空特征缓存"""
        self.feature_cache.clear() 