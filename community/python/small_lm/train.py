"""
Train the tiny Shakespeare language model on CPU and save its weights.

Usage:
    python train.py
"""

import time

import numpy as np
import torch

import small_lm
import utils

MODEL_CONFIG = {"block_size": 64, "n_embd": 64, "n_head": 4, "n_layer": 2}


def main(
    steps: int = 1500, batch_size: int = 32, weights_path: str = "small_lm.pt"
) -> tuple[small_lm.TinyGPT, utils.CharTokenizer, list[float], float]:
    torch.manual_seed(10)
    rng = np.random.default_rng(seed=20)

    # Load the dataset and split it into 90% training and 10% validation data
    text = utils.load_shakespeare()
    tokenizer = utils.CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Dataset: {len(text):,} characters, vocabulary size {tokenizer.vocab_size}")

    model = small_lm.TinyGPT(vocab_size=tokenizer.vocab_size, **MODEL_CONFIG)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")

    # Training runs in plain PyTorch on CPU (training mode of the Q-layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_losses = []
    start = time.time()
    model.train()
    for step in range(steps):
        x_batch, y_batch = utils.get_batch(
            train_data, MODEL_CONFIG["block_size"], batch_size, rng
        )

        optimizer.zero_grad()
        _, loss = model(x_batch, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if step % 250 == 0:
            print(f"Step {step:5d}: loss {loss.item():.3f}")

    print(f"Final training loss: {train_losses[-1]:.3f} ({time.time() - start:.0f} s)")
    print(f"Random guessing would give: {np.log(tokenizer.vocab_size):.3f}")

    # Loss on unseen text
    with torch.no_grad():
        x_val, y_val = utils.get_batch(val_data, MODEL_CONFIG["block_size"], 256, rng)
        _, val_loss = model(x_val, y_val)
        val_loss = val_loss.item()
    print(f"Validation loss: {val_loss:.3f}")

    torch.save({"config": MODEL_CONFIG, "state_dict": model.state_dict()}, weights_path)
    print(f"Saved weights to {weights_path}")

    return model, tokenizer, train_losses, val_loss


if __name__ == "__main__":
    main()
