# Tiny Shakespeare Language Model
> **Community example** contributed by [@jero98772](https://github.com/jero98772) (see [AUTHORS.md](AUTHORS.md)).
> It is not maintained or supported by Q.ANT. See the [community README](../../README.md) for the disclaimer and removal policy.

This project demonstrates text generation with a small character-level transformer (a tiny GPT) executed on the Q.ANT Native Computing Toolkit.

We here use the Q.ANT Native Computing Toolkit to implement PyTorch layers that allow training on CPU / GPU and straightforward evaluation on the NPU.
A basic understanding of PyTorch and transformers is recommended.

## Prerequisites
- The Q.ANT Native Computing Toolkit Python wheel (the CPU backend is sufficient to run the example without NPU hardware).
- Internet access on the first run to download the dataset.

## Overview
Training: train.py \
Trains the language model on CPU and saves the weights to `small_lm.pt`.

Inference: generate.py \
Loads the trained weights and generates text, with the inference running on the NPU.

Layers: small_lm.py \
Contains the Q.ANT layers (`QLinear`, `QReLU`, `QCausalSelfAttention`) and the `TinyGPT` model.

Dataset: Tiny Shakespeare \
A ~1 MB text file with a collection of Shakespeare's plays (https://github.com/karpathy/char-rnn). It is downloaded automatically to `/tmp/shakespeare_files/`.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Train the model once (about a minute on CPU):

```bash
python train.py
```

Then start the generator and type a prompt, e.g. `ROMEO:` or `JULIET:`:

```bash
python generate.py
```

Every prompt produces 200 new characters. Press Enter on an empty prompt to quit.
The prompt may only contain characters that appear in the dataset.

## Testing
PyTest test case:
The file test_small_lm.py trains the model, generates text and checks the training loss, the generated text and the agreement between NPU and CPU inference.

```bash
pytest test_small_lm.py
```

# Description
A two-layer transformer (~100k parameters) is trained on CPU to predict the next character of Shakespeare's text.
After training, the model is switched to `eval()` mode and generates new text, with the following operations executed on the NPU:

| Operation | Executed on | Toolkit function |
|---|---|---|
| Linear layers (Q, K, V projections, MLP, output head) | NPU | `qant.ai.linear_fprop` |
| Attention scores QK^T and weighted sum AV | NPU | `qant.ai.linear_fprop` |
| Attention weights and next-character distribution | NPU | `qant.ai.softmax_fprop` |
| ReLU activation | NPU | `qant.ai.relu_fprop` |
| Embedding lookup, LayerNorm, residual additions | Host | – |
