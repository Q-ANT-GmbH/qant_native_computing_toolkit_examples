import urllib.request
from pathlib import Path

import numpy as np
import torch

torch.manual_seed(42)

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_PATH = Path("/tmp/shakespeare_files/input.txt")


def load_shakespeare() -> str:
    """Download the Tiny Shakespeare dataset (~1 MB) once and return it as a string."""
    if not DATASET_PATH.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            DATASET_PATH.write_bytes(response.read())
    return DATASET_PATH.read_text(encoding="utf-8")


class CharTokenizer:
    """Character-level tokenizer: every distinct character of the text is one token."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = dict(enumerate(self.chars))

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text]

    def decode(self, tokens) -> str:
        return "".join(self.idx_to_char[int(t)] for t in tokens)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random sequences x and the next-character targets y (x shifted by one)."""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y
