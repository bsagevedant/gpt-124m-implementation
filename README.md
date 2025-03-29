# GPT-124M Model Implementation (PyTorch)

This repository contains a PyTorch implementation of the GPT-124M model, following the architecture described in the original GPT paper.

## Features

* **GPT-124M Configuration:** Includes a predefined configuration (`GPT_CONFIG_124M`) for the 124M parameter model.
* **Multi-Head Attention:** Implements the core multi-head attention mechanism with causal masking.
* **Layer Normalization:** Utilizes Layer Normalization for improved training stability.
* **GELU Activation:** Employs the GELU activation function.
* **Feedforward Network:** Includes a feedforward network within each Transformer block.
* **Transformer Blocks:** Stacks multiple Transformer blocks to build the model.
* **Embedding Layers:** Uses token and position embeddings.
* **Output Head:** Linear layer for generating logits.
* **Pre-LayerNorm Architecture:** Implements the "Pre-LayerNorm" transformer structure for better convergence.

## Requirements

* Python 3.x
* PyTorch

## Usage

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install PyTorch:**

    ```bash
    # Install PyTorch according to your system and CUDA setup.
    # See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    ```

3.  **Use the Model:**

    ```python
    import torch
    from gpt_model import GPTModel, GPT_CONFIG_124M

    # Initialize the model
    model = GPTModel(GPT_CONFIG_124M)

    # Example input (batch_size, sequence_length)
    input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 256))

    # Forward pass
    logits = model(input_ids)

    print(logits.shape) # Output logits shape: (2, 256, 50257)
    ```

## Model Architecture

The model consists of the following components:

* **Token Embedding:** Maps input token IDs to dense vectors.
* **Position Embedding:** Adds positional information to the input embeddings.
* **Transformer Blocks:** Stacked layers of multi-head attention and feedforward networks.
* **Layer Normalization:** Normalizes the output of each sub-layer.
* **Output Head:** Linear layer that maps the final hidden states to logits.

## Key Implementation Details

* **Causal Masking:** The multi-head attention mechanism uses a causal mask to prevent the model from attending to future tokens.
* **Pre-LayerNorm:** Layer normalization is applied before the attention and feedforward sub-layers.
* **GELU Activation:** The GELU activation function is used in the feedforward network.
* **Dropout:** Dropout is applied to the embeddings and attention weights.

## Configuration

The `GPT_CONFIG_124M` dictionary defines the model's hyperparameters:

* `vocab_size`: Size of the vocabulary.
* `context_length`: Maximum sequence length.
* `emb_dim`: Embedding dimension.
* `n_heads`: Number of attention heads.
* `n_layers`: Number of Transformer blocks.
* `drop_rate`: Dropout rate.
* `qkv_bias`: Whether to include bias in the query, key, and value projections.

## Notes

* This implementation can be adapted for different GPT model sizes by modifying the `GPT_CONFIG_124M` dictionary.
* For training, you will need to implement a training loop and loss function.
* This code provides the model architecture, and does not include training or data loading pipelines.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

[MIT License](LICENSE)
