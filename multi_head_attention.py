import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # apply the linear transformations to the query, key, and value
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # compute the dot product attention for each head
        query = query.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

        # compute the dot product attention
        attention = torch.matmul(query, key.transpose(2, 3))
        attention = attention / math.sqrt(self.hidden_size // self.num_heads)

        # apply the mask (if any)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        # apply the softmax to compute the weights for each value
        weights = torch.softmax(attention, dim=-1)

        # apply the attention weights to the values
        x = torch.matmul(weights, value)

        # combine the values for all heads and apply the final linear transformation
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        x = self.output_linear(x).transpose(1, 2)

        return x
