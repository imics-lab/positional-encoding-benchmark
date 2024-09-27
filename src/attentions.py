import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class Attention_Rel_Vec(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size / num_heads)))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
                .unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)

        attn = torch.matmul(q, k)
        attn = (attn + Srel) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out

    def skew(self, QEr):
        padded = nn.functional.pad(QEr, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        Srel = reshaped[:, :, 1:, :]
        return Srel


class TUPEMultiHeadAttention(nn.Module):
    def __init__(self, config: TUPEConfig, pos_embed: nn.Module) -> None:
        super().__init__()
        self.max_len = config.max_len
        self.num_heads = config.num_heads
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.bidirectional = config.bidirectional_bias
        self.scale = math.sqrt(2 * config.d_head)

        self.pos_embed = pos_embed
        self.dropout = nn.Dropout(config.dropout)

        # kqv in one pass
        self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        self.relative_bias = config.relative_bias
        if config.relative_bias:
            self.bias = nn.Embedding(config.max_len * 2, config.num_heads)

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        pos_embed = self.pos_embed(seq_len).repeat(batch_size, 1, 1)
        # pos_embed.shape == (batch_size, seq_len, d_model)
        pos_key, pos_query = self.pos_kq(pos_embed).chunk(2, dim=-1)
        pos_key = pos_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # pos_key.shape == (batch_size, num_heads, d_head, seq_len)
        pos_query = pos_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # pos_query.shape == (batch_size, num_heads, seq_len, d_head)
        pos_attn = torch.matmul(pos_query, pos_key)
        # pos_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        tok_key, tok_query, tok_value = self.tok_kqv(x).chunk(3, dim=-1)
        tok_key = tok_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # tok_key.shape == (batch_size, num_heads, d_head, seq_len)
        tok_query = tok_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        tok_value = tok_value.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # tok_qv.shape == (batch_size, num_heads, seq_len, d_head)
        tok_attn = torch.matmul(tok_query, tok_key)
        # tok_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = (tok_attn + pos_attn) / self.scale
        if self.relative_bias:
            relative_positions = get_relative_positions(
                seq_len, self.bidirectional, self.num_buckets, self.max_distance
            ).to(attn.device)
            # relative_positions.shape == (seq_len, seq_len)
            bias = self.bias(relative_positions + self.max_len)
            # bias.shape == (seq_len, seq_len, num_heads)
            bias = bias.permute(2, 0, 1).unsqueeze(0)
            # bias.shape == (1, num_heads, seq_len, seq_len)
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attn, tok_value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.dropout(out)
        return out


class TPS_SelfAttention(nn.Module):
    def __init__(self, num_heads, model_dim, max_len, pow=2, LrEnb=0, LrMo=0, dropout=0.1):
        super(TPS_SelfAttention_Author, self).__init__()
        assert model_dim % num_heads == 0
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.d_k = model_dim // num_heads
        self.pow = pow
        self.LrEnb = LrEnb
        self.LrMo = LrMo
        self.Max_Len = max_len

        # Initialize multi-head attention module
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout)

        # Precompute temporal distance
        t = torch.arange(0, max_len, dtype=torch.float)
        t1 = t.repeat(max_len, 1)
        t2 = t1.permute([1, 0])

        if pow == 2:
            dis1 = torch.exp(-1 * torch.pow((t2 - t1), 2) / 2)
            self.dist = nn.Parameter(-1 * torch.pow((t2 - t1), 2) / 2, requires_grad=False)
        else:
            dis1 = torch.exp(-1 * torch.abs((t2 - t1)))
            self.dist = nn.Parameter(-1 * torch.abs((t2 - t1)), requires_grad=False)

        if LrEnb:
            self.adj1 = nn.Parameter(dis1)  # Learnable temporal weighting

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v expected to have shape: [seq_len, batch_size, embedding_dim]
        
        # Multi-head attention forward pass
        attn_output, attn_weights = self.attention(q, k, v, key_padding_mask=mask)
        
        # Apply Gaussian-based temporal weighting
        seq_len = q.size(0)  # Extract sequence length from q
        batch_size = q.size(1)  # Extract batch size
        num_heads = self.num_heads
        
        # Dynamically compute temporal distance matrix based on seq_len
        t = torch.arange(0, seq_len, dtype=torch.float, device=q.device)
        t1 = t.repeat(seq_len, 1)
        t2 = t1.permute([1, 0])

        if self.pow == 2:
            dist_matrix = torch.exp(-1 * torch.pow((t2 - t1), 2) / 2)
        else:
            dist_matrix = torch.exp(-1 * torch.abs((t2 - t1)))

        # Expand dist_matrix to match the shape of attn_weights [batch_size, seq_len, seq_len]
        expanded_dist = dist_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply Gaussian decay to attention weights
        weighted_attn = attn_weights * expanded_dist

        # Normalize attention scores
        weighted_attn = weighted_attn / weighted_attn.sum(dim=-1, keepdim=True)

        # Reshape `v` to [batch_size, seq_len, model_dim] for correct matrix multiplication
        v = v.permute(1, 0, 2)  # Change `v` from [seq_len, batch_size, model_dim] to [batch_size, seq_len, model_dim]

        # Matrix multiplication for attention output
        output = torch.bmm(weighted_attn, v)  # Apply attention weights to the value matrix

        # Reshape the output back to [seq_len, batch_size, model_dim]
        output = output.permute(1, 0, 2)  # Change back to [seq_len, batch_size, model_dim]

        # Apply dropout to the attention output
        output = self.dropout(output)

        return output, weighted_attn