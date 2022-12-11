import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size):
        super(PositionalEncoding, self).__init__()

        self.hidden_size = hidden_size

        # create the position and distance encodings
        self.position_encoding = nn.Parameter(torch.zeros(1, hidden_size))
        self.distance_encoding = nn.Parameter(torch.zeros(1, hidden_size))

    def forward(self, x):
		# compute the position and distance encodings
		batch_size, seq_len, _ = x.shape
		position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
		div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * -(math.log(10000.0) / self.hidden_size))

		# apply the sin and cos functions to the position and distance encodings
		self.position_encoding[:, 0::2] = torch.sin(position * div_term)
		self.position_encoding[:, 1::2] = torch.cos(position * div_term)
		self.distance_encoding[:, 0::2] = torch.sin(position * div_term)
		self.distance_encoding[:, 1::2] = torch.cos(position * div_term)

		# apply the encodings to the input
		x = x + self.position_encoding
		x = x + self.distance_encoding

		return x


