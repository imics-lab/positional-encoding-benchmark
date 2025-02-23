import torch
import torch.nn as nn
from patch_embedding_layer import TimeSeriesPatchEmbeddingLayer

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_timesteps, in_channels, patch_size, embedding_dim, pos_encoding='fixed', num_transformer_layers=6, num_heads=8, dim_feedforward=128, dropout=0.1, num_classes=2):
        super().__init__()
        
        self.patch_embedding = TimeSeriesPatchEmbeddingLayer(in_channels, patch_size, embedding_dim, input_timesteps, pos_encoding)
        self.num_patches = -(-input_timesteps // patch_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_transformer_layers)
        
        self.ff_layer = nn.Linear(embedding_dim, dim_feedforward)
        self.classifier = nn.Linear(dim_feedforward, num_classes)
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        class_token_output = x[:, 0, :]
        x = self.ff_layer(class_token_output)
        output = self.classifier(x)
        return output
