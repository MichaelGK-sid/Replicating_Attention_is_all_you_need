# Replicating the Transformer Architecture from "Attention Is All You Need"

This repository contains a simplified implementation of the Transformer model from the seminal paper *"Attention Is All You Need"*.  
The goal is to provide a clear, modular, and functional code base for understanding the core components of this powerful architecture, including self-attention, multi-head attention, and positional encoding.

---

## Project Structure

The project is structured into four main files, each with a specific role in building and training the model.

### `encoder.py`
This file contains the core building blocks for the Transformer's Encoder stack.  
It defines the `EncoderLayer` and the main `Encoder` class.  
The encoder processes the input sequence and generates contextualized representations.

### `decoder.py`
This file contains the code for the Transformer's Decoder stack.  
It defines the `DecoderLayer` and the main `Decoder` class.  
The decoder uses both self-attention and cross-attention to generate the output sequence, attending to both its own previous outputs and the encoder's output.

### `transformer_model.py`
This file orchestrates the components from the `encoder.py` and `decoder.py` files.  
It defines the complete Transformer model, which is a sequence-to-sequence architecture that can be used for tasks like machine translation.

### `training_loop.py`
This is the main script that ties everything together. It handles:

- Loading and preprocessing the dataset.  
- Initializing the Transformer model.  
- Defining the optimizer and loss function.  
- Executing the training and evaluation loop.  

This is the file you will run to train your model.

---

## Getting Started

### Prerequisites
To run this code, you will need the following Python libraries. You can install them using pip:

```bash
pip install torch numpy
