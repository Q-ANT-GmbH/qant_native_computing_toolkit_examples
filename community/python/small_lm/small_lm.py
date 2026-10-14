"""
This file contains the implementation of a tiny character-level GPT built from Q.ANT layers.

Like the layers in the Fourier network example, every Q-layer trains on CPU with plain PyTorch
(training mode) and routes its computation through the Q.ANT Native Computing Toolkit in eval mode:

    - QLinear               -> qant.ai.linear_fprop
    - QReLU                 -> qant.ai.relu_fprop
    - QCausalSelfAttention  -> qant.ai.linear_fprop (Q K^T and A V), qant.ai.softmax_fprop

Embedding lookup, LayerNorm and residual additions are executed on the host.
"""

import math

import numpy as np
import torch
from ml_dtypes import bfloat16

import qant_native_computing_toolkit as qant


class QLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dtype == torch.float32, (
            f"QLinear only supports float32 input, but got input with dtype {input.dtype}"
        )

        # Training forward pass
        if self.training:
            # Use the standard linear forward pass from PyTorch for training
            output = torch.nn.functional.linear(input, self.weight, bias=self.bias)
            return output

        # Eval forward pass (no-grad inference)
        else:
            assert input.device.type == "cpu", (
                f"QLinear only supports CPU execution, but got input on device {input.device}"
            )
            # Convert input and weights to bfloat16 numpy arrays
            input_np = input.detach().numpy().astype(bfloat16)
            weights_np = self.weight.data.detach().numpy().astype(bfloat16)

            # Call the linear forward pass on NPU
            output_np = qant.ai.linear_fprop(input_np, weights_np)

            if self.bias is not None:
                bias_np = self.bias.data.detach().numpy().astype(bfloat16)
                return torch.from_numpy((output_np + bias_np).astype(np.float32))
            else:
                return torch.from_numpy(output_np.astype(np.float32))


class QReLU(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Training forward pass
        if self.training:
            return torch.relu(input)

        # Eval forward pass: ReLU on NPU
        else:
            input_np = input.detach().numpy().astype(bfloat16)
            output_np = qant.ai.relu_fprop(input_np)
            return torch.from_numpy(output_np.astype(np.float32))


class QCausalSelfAttention(torch.nn.Module):
    """
    Multi-head causal self-attention. Every token can only attend to itself and previous tokens.

    attention(Q, K, V) = softmax(Q K^T / sqrt(d_head) + causal mask) V
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = QLinear(n_embd, 3 * n_embd)
        self.proj = QLinear(n_embd, n_embd)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape

        # Project to queries, keys and values: (B, n_head, T, head_dim) each
        q, k, v = self.qkv(input).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Training forward pass
        if self.training:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

        # Eval forward pass (no-grad inference)
        else:
            # The toolkit requires C-contiguous inputs, but the head split above returns
            # transposed views. Fold the 1/sqrt(d_head) scaling into the queries.
            q_np = (q / math.sqrt(self.head_dim)).contiguous().numpy().astype(bfloat16)
            k_np = k.contiguous().numpy().astype(bfloat16)
            v_np = v.contiguous().numpy().astype(bfloat16)

            out_np = np.zeros((B, self.n_head, T, self.head_dim), dtype=np.float32)
            for b in range(B):
                for h in range(self.n_head):
                    # linear_fprop computes features @ weights.T, i.e. exactly Q K^T
                    scores = qant.ai.linear_fprop(q_np[b, h], k_np[b, h])

                    # Causal softmax: row t only sees the keys 0..t, the rest stays zero
                    attn = np.zeros((T, T), dtype=bfloat16)
                    for t in range(T):
                        attn[t, : t + 1] = qant.ai.softmax_fprop(
                            np.ascontiguousarray(scores[t, : t + 1])
                        )

                    # A V == A @ (V^T)^T
                    v_t = np.ascontiguousarray(v_np[b, h].T)
                    out_np[b, h] = qant.ai.linear_fprop(attn, v_t).astype(np.float32)
            out = torch.from_numpy(out_np)

        # Merge heads and apply the output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class QMLP(torch.nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.fc = QLinear(n_embd, 4 * n_embd)
        self.act = QReLU()
        self.proj = QLinear(4 * n_embd, n_embd)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.fc(input)))


class Block(torch.nn.Module):
    """Transformer block: pre-LayerNorm attention and MLP, each with a residual connection."""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(n_embd)
        self.attn = QCausalSelfAttention(n_embd, n_head)
        self.ln_2 = torch.nn.LayerNorm(n_embd)
        self.mlp = QMLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(torch.nn.Module):
    """A tiny decoder-only transformer predicting the next character."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
    ) -> None:
        """
        Args:
        ----
            vocab_size: int
                Number of distinct characters
            block_size: int
                Maximum context length in characters
            n_embd: int
                Embedding dimension
            n_head: int
                Number of attention heads
            n_layer: int
                Number of transformer blocks
        """
        super().__init__()
        self.block_size = block_size
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding = torch.nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.ModuleList(
            [Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = QLinear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block size {self.block_size}"
        )

        positions = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss


@torch.no_grad()
def generate(
    model: TinyGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressively sample max_new_tokens characters, starting from the context idx of shape (1, T).

    In eval mode the next-character distribution is computed with the softmax of the NPU.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        last_logits = logits[0, -1] / temperature

        if model.training:
            probs = torch.softmax(last_logits, dim=-1).numpy().astype(np.float64)
        else:
            probs = qant.ai.softmax_fprop(last_logits.numpy().astype(bfloat16))
            probs = probs.astype(np.float64)

        # Renormalize to compensate for bfloat16 rounding before sampling
        next_token = rng.choice(len(probs), p=probs / probs.sum())
        idx = torch.cat([idx, torch.tensor([[next_token]], dtype=idx.dtype)], dim=1)
    return idx
