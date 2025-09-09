import torch
import torch.nn as nn
import math

import encoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = encoder.MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = encoder.MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = encoder.PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 1. Masked self-attention (target attends to itself, no peeking ahead)
        _x = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(_x + self.dropout1(x))

        # 2. Encoder–decoder cross-attention (decoder attends to encoder outputs)
        _x = x
        x = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(_x + self.dropout2(x))

        # 3. Feed-forward
        _x = x
        x = self.ff(x)
        x = self.norm3(_x + self.dropout3(x))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, N=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = encoder.PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Embed target sequence and add position encodings
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        # Pass through stacked decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        return self.norm(x)  # (batch, seq_len, d_model)
