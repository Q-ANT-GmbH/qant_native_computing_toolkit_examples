"""Full assembly of the parts to form the complete network"""

import torch
import qant_native_computing_toolkit.ai as q_ai
import numpy as np
from ml_dtypes import bfloat16


def maxpool(x: np.ndarray, kernel_size, stride=2):
    d = q_ai.maxpool2d_fprop(x, kernel_size, stride, 0)

    return d


def pad(x1: np.ndarray, x2: np.ndarray):
    # input is CHW
    diffY = x2.shape[1] - x1.shape[1]
    diffX = x2.shape[2] - x1.shape[2]

    if diffX > 0 or diffY > 0:
        raise NotImplementedError("Padding not implemented yet")

    return x1, x2


def conv_transpose2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
    d = q_ai.conv_transpose_fprop(x, weight, 0, 2, 1, 0)
    if bias is not None:
        d = q_ai.add_bias_fprop(d, bias)

    return d


def double_conv(x: np.ndarray, block_name, w):
    conv1_w = w[f"{block_name}conv1.weight"]

    bn1_running_mean = w[f"{block_name}norm1.running_mean"]
    bn1_running_var = w[f"{block_name}norm1.running_var"]
    bn1_running_weight = w[f"{block_name}norm1.weight"]
    bn1_running_bias = w[f"{block_name}norm1.bias"]

    conv2_w = w[f"{block_name}conv2.weight"]

    bn2_running_mean = w[f"{block_name}norm2.running_mean"]
    bn2_running_var = w[f"{block_name}norm2.running_var"]
    bn2_running_weight = w[f"{block_name}norm2.weight"]
    bn2_running_bias = w[f"{block_name}norm2.bias"]

    x = q_ai.conv_fprop(x, conv1_w, 1, 1, 1)
    x = q_ai.batchnorm2d_fprop(
        x, bn1_running_mean, bn1_running_var, bn1_running_weight, bn1_running_bias, 0
    )
    x = q_ai.relu_fprop(x)

    x = q_ai.conv_fprop(x, conv2_w, 1, 1, 1)
    x = q_ai.batchnorm2d_fprop(
        x, bn2_running_mean, bn2_running_var, bn2_running_weight, bn2_running_bias, 0
    )
    x = q_ai.relu_fprop(x)

    return x


class QAntUNet:
    """
    A PyTorch-weight compatible implementation of a UNet network
    """

    def __init__(self, w_org, bilinear=False):
        self.w_org = w_org
        self.w = self._preprocess_state_dict(self.w_org)
        self.bilinear = bilinear

    def _preprocess_state_dict(self, w_org):
        w = {}
        for k in w_org.keys():
            w[k] = w_org[k].detach().numpy().astype(bfloat16)

        return w

    def encoder1(self, x: np.ndarray):
        block_name = "encoder1.enc1"
        x = double_conv(x, block_name, self.w)

        return x

    def down1(self, x: np.ndarray):
        block_name = "encoder2.enc2"

        x = maxpool(x, 2)
        x = double_conv(x, block_name, self.w)
        return x

    def down2(self, x: np.ndarray):
        block_name = "encoder3.enc3"

        x = maxpool(x, 2)
        x = double_conv(x, block_name, self.w)
        return x

    def down3(self, x: np.ndarray):
        block_name = "encoder4.enc4"

        x = maxpool(x, 2)
        x = double_conv(x, block_name, self.w)
        return x

    def bottleneck(self, x: np.ndarray):
        block_name = "bottleneck.bottleneck"

        x = maxpool(x, 2)
        x = double_conv(x, block_name, self.w)
        return x

    def up1(self, x1: np.ndarray, x2: np.ndarray):
        double_conv_name = "decoder4.dec4"
        upconv_name = "upconv4"

        if self.bilinear:
            raise NotImplementedError()
        else:
            x1 = conv_transpose2d(
                x1,
                self.w[f"{upconv_name}.weight"],
                self.w[f"{upconv_name}.bias"],
            )

            x1, x2 = pad(x1, x2)
            x = np.concatenate([x1, x2], axis=0)

            x = double_conv(x, double_conv_name, self.w)

            return x

    def up2(self, x1: np.ndarray, x2: np.ndarray):
        double_conv_name = "decoder3.dec3"
        upconv_name = "upconv3"

        if self.bilinear:
            raise NotImplementedError()
        else:
            x1 = conv_transpose2d(
                x1,
                self.w[f"{upconv_name}.weight"],
                self.w[f"{upconv_name}.bias"],
            )

            x1, x2 = pad(x1, x2)
            x = np.concatenate([x1, x2], axis=0)

            x = double_conv(x, double_conv_name, self.w)

            return x

    def up3(self, x1: np.ndarray, x2: np.ndarray):
        double_conv_name = "decoder2.dec2"
        upconv_name = "upconv2"

        if self.bilinear:
            raise NotImplementedError()
        else:
            x1 = conv_transpose2d(
                x1,
                self.w[f"{upconv_name}.weight"],
                self.w[f"{upconv_name}.bias"],
            )

            x1, x2 = pad(x1, x2)
            x = np.concatenate([x1, x2], axis=0)

            x = double_conv(x, double_conv_name, self.w)

            return x

    def up4(self, x1: np.ndarray, x2: np.ndarray):
        double_conv_name = "decoder1.dec1"
        upconv_name = "upconv1"

        if self.bilinear:
            raise NotImplementedError()
        else:
            x1 = conv_transpose2d(
                x1,
                self.w[f"{upconv_name}.weight"],
                self.w[f"{upconv_name}.bias"],
            )

            x1, x2 = pad(x1, x2)
            x = np.concatenate([x1, x2], axis=0)

            x = double_conv(x, double_conv_name, self.w)

            return x

    def outc(self, x: np.ndarray):
        block_name = "conv"
        x = q_ai.conv_fprop(x, self.w[f"{block_name}.weight"], 0, 1, 1)
        return x

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.down1(enc1.copy())
        enc3 = self.down2(enc2.copy())
        enc4 = self.down3(enc3.copy())

        bottleneck = self.bottleneck(enc4)

        dec4 = self.up1(bottleneck, enc4)
        dec3 = self.up2(dec4, enc3)
        dec2 = self.up3(dec3, enc2)
        dec1 = self.up4(dec2, enc1)
        logits = self.outc(dec1)

        return q_ai.sigmoid_fprop(logits)

    def __call__(self, batch: torch.Tensor):
        out_batch = []

        for x in batch:
            _x = x.numpy().astype(bfloat16)
            out = self.forward(_x)

            out = out.astype(np.float32)
            out_batch.append(out)

        return torch.Tensor(np.array(out_batch))
