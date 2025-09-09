import torch
import torch.nn as nn
import torch.optim as optim

import encoder
import decoder



class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        # self.encoder = encoder.Encoder(vocab_size, d_model, n_heads, n_layers)
        # self.decoder = decoder.Decoder(vocab_size, d_model, n_heads, n_layers)
        self.encoder = encoder.Encoder(vocab_size=vocab_size, d_model=d_model, N=n_layers, num_heads=n_heads)
        self.decoder = decoder.Decoder(vocab_size=vocab_size, d_model=d_model, N=n_layers, num_heads=n_heads)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_inp, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt_inp, memory, tgt_mask, src_mask)
        return self.output_layer(out)  # (batch, tgt_len, vocab_size)
