"""
This file contains the implementation of QLinear and QFourier layers.
"""

import math
import torch
import numpy as np
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


class QFourier(torch.nn.Module):
    """
    Single QFourier layer. It computes a Fourier series f(x) for each connection between input and output neurons.

    f(x) = bias_term + sum_{i,k} amplitude_{oik} cos(k x_i + phase_{oik})
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        add_bias: bool,
        noise_std: float | None = 0.04,
    ) -> None:
        """
        Args:
        ----
            in_features: int
                Number of input neurons
            out_features: int
                Number of output neurons
            grid_size: int
                Number of Fourier terms in the layer
            add_bias: bool
                Whether to add a bias term
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.noise_std = noise_std

        # Initialize amplitude parameters
        amplitude_init_std = math.sqrt(2 / ((in_features + out_features) * grid_size))
        self.amplitude = torch.nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * amplitude_init_std
        )

        # Initialize phase parameters
        self.phase = torch.nn.Parameter(
            torch.empty(out_features, in_features, grid_size).uniform_(
                -math.pi / 2, math.pi / 2
            )
        )

        # Initialize bias if needed
        if add_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Create frequency index k as a buffer (not trainable)
        k = torch.arange(1, grid_size + 1).float()
        self.register_buffer("k", k)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dtype == torch.float32, (
            f"QFourier only supports float32 input, but got input with dtype {input.dtype}"
        )

        # Training forward pass
        if self.training:
            # Compute the Fourier series output using broadcasting
            output = torch.sum(
                self.amplitude[None, :, :, :]
                * torch.cos(
                    input[:, None, :, None] * self.get_buffer("k")[None, None, None, :]
                    + self.phase[None, :, :, :]
                ),
                dim=(2, 3),
            )

            if self.noise_std is not None:
                output += torch.randn_like(output) * self.noise_std

            if self.bias is not None:
                return output + self.bias[None, :]
            else:
                return output

        # Eval forward pass (no-grad inference)
        else:
            assert input.device.type == "cpu", (
                f"QFourier only supports execution from host, but got input on device {input.device}"
            )

            # Convert input, amplitude, phase, and k to bfloat16 numpy arrays
            input_np = input.detach().numpy().astype(bfloat16)
            amplitude_np = self.amplitude.data.detach().numpy().astype(bfloat16)
            phase_np = self.phase.data.detach().numpy().astype(bfloat16)
            k_np = self.get_buffer("k").numpy().astype(bfloat16)

            # Call the Fourier forward pass on NPU
            output_np = qant.ai.calc_kan_layer_fprop(
                input_np, phase_np, amplitude_np, k_np
            )

            if self.bias is not None:
                bias_np = self.bias.data.detach().numpy().astype(bfloat16)
                return torch.from_numpy((output_np + bias_np).astype(np.float32))
            else:
                return torch.from_numpy(output_np.astype(np.float32))
