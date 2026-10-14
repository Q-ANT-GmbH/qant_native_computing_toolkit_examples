"""
Generate Shakespeare-like text with a trained model, running inference on the NPU.

Usage:
    python generate.py

Type a prompt (e.g. "ROMEO:") and press Enter. An empty prompt quits.
"""

import time

import numpy as np
import torch

import qant_native_computing_toolkit as qant
import small_lm
import utils


def load_model(
    weights_path: str = "small_lm.pt",
) -> tuple[small_lm.TinyGPT, utils.CharTokenizer]:
    tokenizer = utils.CharTokenizer(utils.load_shakespeare())
    checkpoint = torch.load(weights_path)
    model = small_lm.TinyGPT(vocab_size=tokenizer.vocab_size, **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, tokenizer


def generate_text(
    model: small_lm.TinyGPT,
    tokenizer: utils.CharTokenizer,
    prompt: str,
    length: int,
    temperature: float = 0.8,
    seed: int | None = None,
) -> str:
    unknown = set(prompt) - set(tokenizer.chars)
    if unknown:
        raise ValueError(f"Prompt contains characters not in the vocabulary: {unknown}")

    # eval() routes the Q-layers through the Q.ANT Native Computing Toolkit
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    qant.generic.init_npu(0)
    tokens = small_lm.generate(
        model, context, length, np.random.default_rng(seed), temperature
    )
    qant.generic.release_npu(0)

    return tokenizer.decode(tokens[0])


def main(
    length: int = 200, temperature: float = 0.8, weights_path: str = "small_lm.pt"
) -> None:
    model, tokenizer = load_model(weights_path)

    while True:
        try:
            prompt = input("\nPrompt (empty to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break

        try:
            start = time.time()
            text = generate_text(model, tokenizer, prompt, length, temperature)
        except ValueError as error:
            print(error)
            continue

        print(f"\n{text}\n")
        print(f"(generated {length} characters in {time.time() - start:.1f} s)")


if __name__ == "__main__":
    main()
