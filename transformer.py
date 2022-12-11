import torch
import torch.nn as nn

import math 

from decoder import DecoderLayer
from encoder import EncoderLayer
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention


'''
This is just one possible way to implement the Transformer in PyTorch, and you may need to make some modifications to the code to suit your specific use case.
'''

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.output_linear = nn.Linear(hidden_size, input_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])

    def forward(self, src, tgt, src_mask, tgt_mask):
        # apply the linear transformation to the input
        x = self.input_linear(src)
        x = x * math.sqrt(self.hidden_size)
        x = self.positional_encoding(x)

       # compute the mask for the encoder self-attention
        encoder_mask = src_mask.unsqueeze(1).unsqueeze(2)

        # apply the encoder layers
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)

        # compute the mask for the decoder self-attention
        decoder_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

        # apply the decoder layers
        for layer in self.decoder_layers:
            x = layer(x, decoder_mask)

        # apply the final linear transformation to the output
        x = self.output_linear(x)

        return x
