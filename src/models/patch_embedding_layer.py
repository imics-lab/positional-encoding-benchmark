import torch
import torch.nn as nn
from positional_encodings import get_pos_encoder

class TimeSeriesPatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim, input_timesteps, pos_encoding='fixed'):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        self.num_patches = -(-input_timesteps // patch_size)
        self.padding = (self.num_patches * patch_size) - input_timesteps

        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token_embeddings = nn.Parameter(
            torch.randn((1, 1, embedding_dim), requires_grad=True)
        )
        
        pos_encoder_class = get_pos_encoder(pos_encoding)
        self.position_embeddings = pos_encoder_class(embedding_dim, dropout=0.1, max_len=input_timesteps)

    def forward(self, x):
        if self.padding > 0:
            x = nn.functional.pad(x, (0, 0, 0, self.padding))

        x = x.permute(0, 2, 1)
        conv_output = self.conv_layer(x)
        conv_output = conv_output.permute(0, 2, 1)

        batch_size = x.shape[0]
        class_tokens = self.class_token_embeddings.expand(batch_size, -1, -1)
        output = torch.cat((class_tokens, conv_output), dim=1)

        output = self.position_embeddings(output)

        return output
