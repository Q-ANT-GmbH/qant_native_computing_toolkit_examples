"""
This file contains the implementation of ResNet-18 using the Q.ANT Native Computing Toolkit for image classification. The implementation includes the necessary components such as convolutional layers, batch normalization, and fully connected layers, all optimized for the Q.ANT architecture.
"""

from typing import Mapping, Any
import torch
import numpy as np
import qant_native_computing_toolkit.ai as q_ai
import ml_dtypes


def downsample(
    x: np.ndarray,
    conv_weight,
    bn_running_mean,
    bn_running_var,
    bn_weight,
    bn_bias,
    stride=1,
):
    """Downsample the input using a convolutional layer followed by batch normalization."""
    out = q_ai.conv_fprop(x, conv_weight, padding=0, stride=stride, dilation=1)
    out = q_ai.batchnorm2d_fprop(
        out, bn_running_mean, bn_running_var, bn_weight, bn_bias, 1e-5
    )
    return out


def basic_block(
    x: np.ndarray,
    w,
    layer_name,
    stride: int = 1,
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

    out = q_ai.conv_fprop(
        x, w[f"{layer_name}.conv1.weight"], padding=1, stride=stride, dilation=1
    )
    out = q_ai.batchnorm2d_fprop(
        out,
        w[f"{layer_name}.bn1.running_mean"],
        w[f"{layer_name}.bn1.running_var"],
        w[f"{layer_name}.bn1.weight"],
        w[f"{layer_name}.bn1.bias"],
        1e-5,
    )

    out = q_ai.relu_fprop(out)

    out = q_ai.conv_fprop(
        out, w[f"{layer_name}.conv2.weight"], padding=1, stride=1, dilation=1
    )
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


class QantResNet18:
    """
    This class implements the ResNet-18 architecture using the Q.ANT Native Computing Toolkit. It includes methods for each layer of the network, as well as a forward method to process input data through the network. The weights for the layers are preprocessed and stored in a format suitable for the Q.ANT toolkit.
    """

    def __init__(self, w: Mapping[str, Any]) -> None:
        self.w_org = w
        self.w = self._preprocess_state_dict(self.w_org)

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

    def _preprocess_state_dict(self, w_org):
        w = {}
        for k in w_org.keys():
            w[k] = w_org[k].detach().numpy().astype(ml_dtypes.bfloat16)

        return w

    def conv1(self, x: np.ndarray) -> np.ndarray:
        """First convolutional layer of ResNet-18."""
        return q_ai.conv_fprop(
            x, self.w["conv1.weight"], stride=2, padding=3, dilation=1
        )

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
        x = q_ai.linear_fprop(x, self.w["fc.weight"])
        return q_ai.add_bias_fprop(x, self.w["fc.bias"])

    def layer1(self, x: np.ndarray) -> np.ndarray:
        """First layer of ResNet-18, consisting of two basic blocks."""
        stride = 1
        planes = 64
        block_expansion = 1
        previous_dilation = self.dilation
        blocks = 2
        layer = "layer1"

        downsampling_func = None
        if stride != 1 or self.inplanes != planes * block_expansion:

            def downsampling_func(x):
                return downsample(
                    x,
                    self.w[f"{layer}.0.downsample.0.weight"],
                    self.w[f"{layer}.0.downsample.1.running_mean"],
                    self.w[f"{layer}.0.downsample.1.running_var"],
                    self.w[f"{layer}.0.downsample.1.weight"],
                    self.w[f"{layer}.0.downsample.1.bias"],
                    stride=stride,
                )

        x = basic_block(
            x,
            self.w,
            f"{layer}.0",
            stride,
            self.groups,
            self.base_width,
            previous_dilation,
            downsampling_func,
        )

        for i in range(1, blocks):
            x = basic_block(
                x,
                self.w,
                f"{layer}.{i}",
                1,
                self.groups,
                self.base_width,
                previous_dilation,
            )

        return x

    def layer2(self, x: np.ndarray) -> np.ndarray:
        """Second layer of ResNet-18, consisting of two basic blocks."""
        stride = 2
        planes = 128
        block_expansion = 1
        previous_dilation = self.dilation
        blocks = 2
        layer = "layer2"

        downsampling_func = None
        if stride != 1 or self.inplanes != planes * block_expansion:

            def downsampling_func(x):
                return downsample(
                    x,
                    self.w[f"{layer}.0.downsample.0.weight"],
                    self.w[f"{layer}.0.downsample.1.running_mean"],
                    self.w[f"{layer}.0.downsample.1.running_var"],
                    self.w[f"{layer}.0.downsample.1.weight"],
                    self.w[f"{layer}.0.downsample.1.bias"],
                    stride=stride,
                )

        x = basic_block(
            x,
            self.w,
            f"{layer}.0",
            stride,
            self.groups,
            self.base_width,
            previous_dilation,
            downsampling_func,
        )

        for i in range(1, blocks):
            x = basic_block(
                x,
                self.w,
                f"{layer}.{i}",
                1,
                self.groups,
                self.base_width,
                previous_dilation,
            )

        return x

    def layer3(self, x: np.ndarray) -> np.ndarray:
        """Third layer of ResNet-18, consisting of two basic blocks."""
        stride = 2
        planes = 256
        block_expansion = 1
        previous_dilation = self.dilation
        blocks = 2
        layer = "layer3"

        downsampling_func = None
        if stride != 1 or self.inplanes != planes * block_expansion:

            def downsampling_func(x):
                return downsample(
                    x,
                    self.w[f"{layer}.0.downsample.0.weight"],
                    self.w[f"{layer}.0.downsample.1.running_mean"],
                    self.w[f"{layer}.0.downsample.1.running_var"],
                    self.w[f"{layer}.0.downsample.1.weight"],
                    self.w[f"{layer}.0.downsample.1.bias"],
                    stride=stride,
                )

        x = basic_block(
            x,
            self.w,
            f"{layer}.0",
            stride,
            self.groups,
            self.base_width,
            previous_dilation,
            downsampling_func,
        )

        for i in range(1, blocks):
            x = basic_block(
                x,
                self.w,
                f"{layer}.{i}",
                1,
                self.groups,
                self.base_width,
                previous_dilation,
            )

        return x

    def layer4(self, x: np.ndarray) -> np.ndarray:
        """Fourth layer of ResNet-18, consisting of two basic blocks."""
        stride = 2
        planes = 512
        block_expansion = 1
        previous_dilation = self.dilation
        blocks = 2
        layer = "layer4"

        downsampling_func = None
        if stride != 1 or self.inplanes != planes * block_expansion:

            def downsampling_func(x):
                return downsample(
                    x,
                    self.w[f"{layer}.0.downsample.0.weight"],
                    self.w[f"{layer}.0.downsample.1.running_mean"],
                    self.w[f"{layer}.0.downsample.1.running_var"],
                    self.w[f"{layer}.0.downsample.1.weight"],
                    self.w[f"{layer}.0.downsample.1.bias"],
                    stride=stride,
                )

        x = basic_block(
            x,
            self.w,
            f"{layer}.0",
            stride,
            self.groups,
            self.base_width,
            previous_dilation,
            downsampling_func,
        )

        for i in range(1, blocks):
            x = basic_block(
                x,
                self.w,
                f"{layer}.{i}",
                1,
                self.groups,
                self.base_width,
                previous_dilation,
            )

        return x

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


def to_class_prob(output):
    """Convert the output of the network to class probabilities."""
    output = output.numpy().astype(ml_dtypes.bfloat16)
    probabilities = q_ai.softmax_fprop(output.squeeze()).squeeze()
    return torch.from_numpy(probabilities.astype(np.float32))
