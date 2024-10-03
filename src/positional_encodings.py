import torch
import torch.nn as nn
import math

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
    

def _get_relative_position_bucket(
    relative_position, bidirectional, num_buckets, max_distance
):
    """
    from https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(
            relative_position, torch.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large,
        torch.full_like(relative_postion_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(
        is_small, relative_position, relative_postion_if_large
    )
    return relative_buckets


def get_relative_positions(
    seq_len, bidirectional=True, num_buckets=32, max_distance=128
):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    relative_positions = _get_relative_position_bucket(
        x - y, bidirectional, num_buckets, max_distance
    )
    return relative_positions    

def get_pos_encoder(pos_encoding):
    if pos_encoding == 'fixed':
        return FixedPositionalEncoding
    elif pos_encoding == 'learned':
        return LearnedPositionalEncoding
    elif pos_encoding == 'tape':
        return tAPE
    elif pos_encoding == 'absolute':
        return AbsolutePositionalEncoding
    else:
        raise ValueError(f"Unknown positional encoding type: {pos_encoding}")
    


class SineSPE(nn.Module):
    def __init__(self, in_features, max_len=512):
        super(SineSPE, self).__init__()
        self.in_features = in_features
        self.max_len = max_len
        self.position = nn.Parameter(torch.zeros(1, max_len, in_features))
        self.register_buffer('sine', self._generate_sine_encoding())

    def _generate_sine_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.in_features, 2).float() * -(math.log(10000.0) / self.in_features))
        encoding = torch.zeros(self.max_len, self.in_features)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, seq_len):
        return self.sine[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, in_features)


class ConvSPE(nn.Module):
    def __init__(self, num_heads, in_features, kernel_size=3, num_realizations=1):
        super(ConvSPE, self).__init__()
        padding = kernel_size // 2  # This ensures that the output size matches the input size if stride=1

        # Define a 1D convolutional layer
        self.conv = nn.Conv1d(in_features, in_features, kernel_size=kernel_size, padding=padding)

        self.num_heads = num_heads
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.num_realizations = num_realizations

    def forward(self, x):
        # x should be of shape (batch_size, seq_len, in_features)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, in_features, seq_len)
        x = self.conv(x)  # Apply convolution
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, in_features)
        return x    
    

# Temporal Positional Encoding (T-PE)
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=896):  # Assuming 896 timesteps
        super(TemporalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
    

# Variable Positional Encoding for handling multivariate data
class VariablePositionalEncoding(nn.Module):
    def __init__(self, d_model, num_variables):
        super(VariablePositionalEncoding, self).__init__()
        self.variable_embedding = nn.Embedding(num_variables, d_model)

    def forward(self, x, variable_idx):
        variable_embed = self.variable_embedding(variable_idx)
        return x + variable_embed.unsqueeze(0)
