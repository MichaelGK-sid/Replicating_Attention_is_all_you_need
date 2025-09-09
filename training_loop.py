
import torch
import torch.nn as nn
import torch.optim as optim

import transformer_model

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
VOCAB_SIZE = 100 # example
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6


def shift_right(tgt, bos_idx=BOS_IDX):
    bos = torch.full((tgt.size(0), 1), bos_idx, dtype=torch.long, device=tgt.device)
    return torch.cat([bos, tgt[:, :-1]], dim=1)


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_padding_mask(seq, pad_idx=PAD_IDX):
    # shape: (batch, 1, 1, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


# Example data
src = torch.tensor([[5, 23, 8, 92, 23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8,23, 8, EOS_IDX, PAD_IDX, PAD_IDX]])   # English sentence
tgt = torch.tensor([[4, 15, 38, 77,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38,15, 38, EOS_IDX, PAD_IDX, PAD_IDX]])  # French sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = transformer_model.Transformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=1e-10)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    
    # Move to device
    src, tgt = src.to(device), tgt.to(device)
    
    # Prepare shifted target input for decoder
    tgt_inp = shift_right(tgt)
    
    # Masks
    src_mask = create_padding_mask(src)
    tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1)).to(device)
    
    # Forward pass
    logits = model(src, tgt_inp, src_mask, tgt_mask)
    
    # Compute loss (reshape: (batch*tgt_len, vocab_size))
    print("logits", logits.view( -1, VOCAB_SIZE).shape, logits.view(-1, VOCAB_SIZE).dtype)
    print("tgt", tgt.view(-1).shape, tgt.view(-1).dtype)
    loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")
