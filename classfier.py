import numpy as np
import torch
import torch.nn as nn
import cv2


class MultiClassTransformer(nn.Module):
    def __init__(self, num_points=17, d_model=64, num_heads=8, num_layers=3, num_classes=3):
        super(MultiClassTransformer, self).__init__()
        
        self.embedding = nn.Linear(2, d_model)  # 각 (x, y) 포인트를 d_model 임베딩으로 변환
        self.positional_encoding = self._generate_positional_encoding(num_points, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(num_points * d_model, num_classes)  # 최종 출력 레이어
        
    def _generate_positional_encoding(self, num_points, d_model):
        position = torch.arange(0, num_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(num_points, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch_size, num_points, 2)
        x = self.embedding(x)  # 임베딩
        x = x + self.positional_encoding  # 위치 인코딩 추가
        x = x.permute(1, 0, 2)  # Transformer 인코더에 맞게 차원 변경
        
        encoded = self.transformer_encoder(x)  # Transformer 인코더에 입력
        encoded = encoded.permute(1, 0, 2).reshape(x.shape[1], -1)  # 평탄화
        
        out = self.fc_out(encoded)  # 분류 레이어
        return out