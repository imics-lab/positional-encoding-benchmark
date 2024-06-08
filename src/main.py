import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from load_data import get_dataset


# Positional Encodings
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == 'fixed':
        return FixedPositionalEncoding
    elif pos_encoding == 'learned':
        return LearnedPositionalEncoding
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_encoding}")

# Activation Function
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Invalid activation function: {activation}")

# Custom Transformer Encoder Layer with Batch Normalization
class TransformerBatchNormEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = src.transpose(0, 1).transpose(1, 2)
        src = self.norm1(src)
        src = src.transpose(1, 2).transpose(0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.transpose(0, 1).transpose(1, 2)
        src = self.norm2(src)
        src = src.transpose(1, 2).transpose(0, 1)
        return src

# Main Transformer Encoder Model
class TSTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        output = self.output_layer(output)
        return output

ds_list = ["UniMiB SHAR",
           "UCI HAR",
           "TWristAR",
           "Leotta_2021",
           "Gesture Phase Segmentation"
           ]
for i in ds_list:
    dataset = i
    print("**** ", dataset, " ****")

X_train, y_train, X_valid, y_valid, X_test, y_test, k_size, EPOCHS, t_names = get_dataset(dataset)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            padding_masks = (inputs != 0).any(dim=-1)
            outputs = model(inputs, padding_masks)
            outputs = outputs.view(-1, model.feat_dim)  # Flatten output to (batch_size * seq_length, feat_dim)
            labels = labels.view(-1)  # Flatten labels to (batch_size * seq_length)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Evaluation Function
def evaluate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            padding_masks = (inputs != 0).any(dim=-1)
            outputs = model(inputs, padding_masks)
            outputs = outputs.view(-1, model.feat_dim)  # Flatten output to (batch_size * seq_length, feat_dim)
            labels = labels.view(-1)  # Flatten labels to (batch_size * seq_length)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
    accuracy = correct / (len(val_loader.dataset) * val_loader.dataset.tensors[0].shape[1])
    return total_loss / len(val_loader), accuracy

# Initialize and Train Models with Different Positional Encodings
models = {
    'fixed': TSTransformerEncoder(feat_dim=32, max_len=50, d_model=128, n_heads=8, num_layers=6, dim_feedforward=512, pos_encoding='fixed'),
    'learned': TSTransformerEncoder(feat_dim=32, max_len=50, d_model=128, n_heads=8, num_layers=6, dim_feedforward=512, pos_encoding='learned'),
}

criterion = nn.CrossEntropyLoss()
results = {}

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    results[name] = {'val_loss': val_loss, 'val_accuracy': val_accuracy}

print(results)
