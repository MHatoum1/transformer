# PyTorch Implementation of "Attention Is All You Need"

This is a PyTorch implementation of the Transformer model from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The Transformer model uses self-attention mechanisms to process input sequences and generate output sequences.

## Requirements

- PyTorch 1.0 or higher
- Python 3.6 or higher

### Modules
The `transformer.py` file contains an implementation of the Transformer model in PyTorch. The `Transformer` class extends the `nn.Module` class from PyTorch, and it uses the following components:

- `EncoderLayer`: A single layer of the encoder, which uses a multi-headed self-attention mechanism to compute a weighted sum of the input sequence.
- `DecoderLayer`: A single layer of the decoder, which uses a multi-headed self-attention mechanism to compute a weighted sum of the input sequence, and a multi-headed attention mechanism to compute a weighted sum of the output of the encoder.
- `MultiHeadAttention`: A multi-headed attention mechanism that allows the model to attend to different parts of the input sequence at the same time.
- `PositionalEncoding`: A learnable encoding of the position of each element in the input sequence, which is added to the input before it is passed through the encoder and decoder layers.

To use the `Transformer` class, you will need to instantiate an instance of the class and call its `forward` method. The `forward` method takes four arguments:

- `src`: The input sequence, which should be a tensor of shape `(batch_size, sequence_length, input_size)`.
- `tgt`: The target sequence, which should be a tensor of shape `(batch_size, sequence_length, input_size)`.
- `src_mask`: A mask for the input sequence, which should be a tensor of shape `(batch_size, 1, sequence_length)`.
- `tgt_mask`: A mask for the target sequence, which should be a tensor of shape `(batch_size, sequence_length, sequence_length)`.

The `forward` method will return the output of the model, which will have the same shape as the input.


### References

* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need," arXiv:1706.03762, 2017. ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)