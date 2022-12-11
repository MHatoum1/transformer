import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(EncoderLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x, mask=None):
        # apply the self-attention
        x = self.self_attention(x, x, x, mask)

        # apply the position-wise feedforward network
        x = self.linear1(x)
        x = self.linear2(x)

        return x
