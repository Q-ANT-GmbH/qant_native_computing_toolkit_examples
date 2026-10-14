"""
This file contains the implementation of ResNet-18 using the Q.ANT Native Computing Toolkit for image classification.
Unlike qant_resnet.py, the weights of every conv/fc layer are uploaded to NPU memory once, when the model is constructed, and the
resulting handles are kept for the lifetime of the model, which avoids re-uploading the full set of weights for every image.
"""

from collections.abc import Mapping
from typing import Any

import ml_dtypes
import numpy as np
import qant_native_computing_toolkit.ai as q_ai
import qant_native_computing_toolkit.generic as q_generic
import torch


class MemLinearLayer:
    """A single matrix-vector multiplication executed via the NPU's memory-management API. The weight
    matrix is uploaded to NPU memory once, at construction, and the feature/result memory is
    allocated once and reused for every call."""

    def __init__(self, weights: np.ndarray, device_id: int = 0):
        out_features, in_features = weights.shape

        self.w_handle = q_generic.upload(
            weights, q_generic.DataRole.Weights, device_id=device_id
        )
        self.f_handle = q_generic.allocate(
            1, in_features, q_generic.DataRole.Features, device_id=device_id
        )
        self.r_handle = q_generic.allocate(
            1, out_features, q_generic.DataRole.Features, device_id=device_id
        )

    def __call__(self, features: np.ndarray) -> np.ndarray:
        q_generic.upload_into(features, self.f_handle)
        q_ai.linear_fprop_preloaded(self.f_handle, self.w_handle, self.r_handle)
        return q_generic.download(self.r_handle)


class MemConvLayer:
    """A convolution whose kernels are uploaded to NPU memory once, at construction, and reused for every call."""

    def __init__(
        self, kernels: np.ndarray, padding: int, stride: int, device_id: int = 0
    ):
        self.handle = q_ai.upload_conv_weights(kernels, device_id=device_id)
        self.padding = padding
        self.stride = stride

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return q_ai.conv_fprop_preloaded_weights(
            x, self.handle, padding=self.padding, stride=self.stride, dilation=1
        )


def downsample_with_mem(
    x: np.ndarray,
    conv_layer: MemConvLayer,
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
):
    """Downsample the input using a convolutional layer followed by batch normalization."""
    out = conv_layer(x)
    out = q_ai.batchnorm2d_fprop(
        out, bn_running_mean, bn_running_var, bn_weight, bn_bias, 1e-5
    )
    return out


def basic_block_with_mem(
    x: np.ndarray,
    w,
    conv_layers: Mapping[str, MemConvLayer],
    layer_name,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    downsample=None,
) -> np.ndarray:
    """Basic block of ResNet-18, consisting of two convolutional layers with batch normalization and a skip connection."""

    if groups != 1 or base_width != 64:
        raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1:
        raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    identity = x.copy()

    out = conv_layers[f"{layer_name}.conv1.weight"](x)
    out = q_ai.batchnorm2d_fprop(
        out,
        w[f"{layer_name}.bn1.running_mean"],
        w[f"{layer_name}.bn1.running_var"],
        w[f"{layer_name}.bn1.weight"],
        w[f"{layer_name}.bn1.bias"],
        1e-5,
    )

    out = q_ai.relu_fprop(out)

    out = conv_layers[f"{layer_name}.conv2.weight"](out)
    out = q_ai.batchnorm2d_fprop(
        out,
        w[f"{layer_name}.bn2.running_mean"],
        w[f"{layer_name}.bn2.running_var"],
        w[f"{layer_name}.bn2.weight"],
        w[f"{layer_name}.bn2.bias"],
        1e-5,
    )

    if downsample is not None:
        identity = downsample(identity)

    out += identity
    out = q_ai.relu_fprop(out)

    return out


class QantResNet18WithMem:
    """
    This class implements the ResNet-18 architecture using the Q.ANT Native Computing Toolkit's explicit
    memory-management API.
    """

    # (layer name, output planes, stride of the first block) for each of the four ResNet-18 stages.
    _LAYER_SPECS = (
        ("layer1", 64, 1),
        ("layer2", 128, 2),
        ("layer3", 256, 2),
        ("layer4", 512, 2),
    )

    def __init__(self, w: Mapping[str, Any], device_id: int = 0) -> None:
        self.w_org = w
        self.w = self._preprocess_state_dict(self.w_org)
        self.device_id = device_id

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv_layers = self._build_conv_layers(device_id)
        self.fc_layer = MemLinearLayer(self.w["fc.weight"], device_id=device_id)

    def _preprocess_state_dict(self, w_org):
        w = {}
        for k in w_org:
            w[k] = w_org[k].detach().numpy().astype(ml_dtypes.bfloat16)

        return w

    def _build_conv_layers(self, device_id: int):
        """Upload every convolution's kernels to NPU memory once, keyed by their state-dict weight name."""
        conv_layers = {}

        conv_layers["conv1.weight"] = MemConvLayer(
            self.w["conv1.weight"], padding=3, stride=2, device_id=device_id
        )

        block_expansion = 1
        for layer, planes, stride in self._LAYER_SPECS:
            has_downsample = stride != 1 or self.inplanes != planes * block_expansion

            for b in range(2):
                block_stride = stride if b == 0 else 1
                conv_layers[f"{layer}.{b}.conv1.weight"] = MemConvLayer(
                    self.w[f"{layer}.{b}.conv1.weight"],
                    padding=1,
                    stride=block_stride,
                    device_id=device_id,
                )
                conv_layers[f"{layer}.{b}.conv2.weight"] = MemConvLayer(
                    self.w[f"{layer}.{b}.conv2.weight"],
                    padding=1,
                    stride=1,
                    device_id=device_id,
                )

            if has_downsample:
                conv_layers[f"{layer}.0.downsample.0.weight"] = MemConvLayer(
                    self.w[f"{layer}.0.downsample.0.weight"],
                    padding=0,
                    stride=stride,
                    device_id=device_id,
                )

        return conv_layers

    def conv1(self, x: np.ndarray) -> np.ndarray:
        """First convolutional layer of ResNet-18."""
        return self.conv_layers["conv1.weight"](x)

    def bn1(self, x: np.ndarray) -> np.ndarray:
        """First batch normalization layer of ResNet-18."""
        x = q_ai.batchnorm2d_fprop(
            x,
            self.w["bn1.running_mean"],
            self.w["bn1.running_var"],
            self.w["bn1.weight"],
            self.w["bn1.bias"],
            1e-5,
        )
        return x

    def maxpool(self, x: np.ndarray) -> np.ndarray:
        """Max pooling layer of ResNet-18."""
        x = q_ai.maxpool2d_fprop(x, 3, 2, 1)
        return x

    def avgpool(self, x: np.ndarray) -> np.ndarray:
        """Average pooling layer of ResNet-18."""
        x = q_ai.adaptive_avgpool2d_fprop(x, 1, 1)
        return x

    def fc(self, x: np.ndarray) -> np.ndarray:
        """Fully connected layer of ResNet-18."""
        x = self.fc_layer(x.reshape(1, -1))
        x = x.reshape(-1)
        return q_ai.add_bias_fprop(x, self.w["fc.bias"])

    def _forward_layer(self, x: np.ndarray, layer: str) -> np.ndarray:
        """Forward pass through one of the four ResNet-18 stages, each consisting of two basic blocks."""
        downsampling_func = None
        if f"{layer}.0.downsample.0.weight" in self.conv_layers:

            def downsampling_func(x):
                return downsample_with_mem(
                    x,
                    self.conv_layers[f"{layer}.0.downsample.0.weight"],
                    self.w[f"{layer}.0.downsample.1.running_mean"],
                    self.w[f"{layer}.0.downsample.1.running_var"],
                    self.w[f"{layer}.0.downsample.1.weight"],
                    self.w[f"{layer}.0.downsample.1.bias"],
                )

        x = basic_block_with_mem(
            x,
            self.w,
            self.conv_layers,
            f"{layer}.0",
            self.groups,
            self.base_width,
            self.dilation,
            downsampling_func,
        )
        x = basic_block_with_mem(
            x,
            self.w,
            self.conv_layers,
            f"{layer}.1",
            self.groups,
            self.base_width,
            self.dilation,
        )

        return x

    def layer1(self, x: np.ndarray) -> np.ndarray:
        """First layer of ResNet-18, consisting of two basic blocks."""
        return self._forward_layer(x, "layer1")

    def layer2(self, x: np.ndarray) -> np.ndarray:
        """Second layer of ResNet-18, consisting of two basic blocks."""
        return self._forward_layer(x, "layer2")

    def layer3(self, x: np.ndarray) -> np.ndarray:
        """Third layer of ResNet-18, consisting of two basic blocks."""
        return self._forward_layer(x, "layer3")

    def layer4(self, x: np.ndarray) -> np.ndarray:
        """Fourth layer of ResNet-18, consisting of two basic blocks."""
        return self._forward_layer(x, "layer4")

    def forward(self, x: np.ndarray):
        """Forward pass through the ResNet-18 architecture."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = q_ai.relu_fprop(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten()
        x = self.fc(x)
        return x

    def __call__(self, batch: torch.Tensor):
        """Call method to process a batch of input data through the ResNet-18 architecture. The input batch is expected to be a PyTorch tensor, which is converted to a NumPy array and processed through the forward method. The output is then converted back to a PyTorch tensor."""
        out_batch = []

        for x in batch:
            _x = (x.numpy()).astype(ml_dtypes.bfloat16)
            out = self.forward(_x)

            out = out.astype(np.float32)
            out_batch.append(out)

        return torch.from_numpy(np.array(out_batch))
