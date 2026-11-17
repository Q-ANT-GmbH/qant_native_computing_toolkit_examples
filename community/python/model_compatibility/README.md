# Check and run an existing PyTorch model on the NPU
> **Community example** contributed by [@alijnmerchant21](https://github.com/alijnmerchant21) (see [AUTHORS.md](AUTHORS.md)).
> It is not maintained or supported by Q.ANT. See the [community README](../../README.md) for the disclaimer and removal policy.

You already have a trained PyTorch model. This example tells you which parts of it can run on the NPU, how much data crosses the PCIe link in one forward pass, and how much accuracy you lose in bfloat16, without rewriting the model.

The official examples rebuild known networks by hand, and `small_lm` designs a model around toolkit layers from the start. This example covers the step in between: it starts from an unmodified `nn.Module`.

## Prerequisites
- The Q.ANT Native Computing Toolkit Python wheel (the CPU backend is enough, no NPU hardware needed).
- Internet access on the first run, to download MNIST to `/tmp/mnist_files/`.

## Overview
Converter: npu_compat.py \
Traces any `torch.fx`-traceable model, maps each node to a toolkit function, uploads weights once, and runs the rest in PyTorch.

Example model: small_cnn.py \
An ordinary MNIST CNN with convolution, batch norm, ReLU, max pooling and linear layers. The first run trains it for one epoch on the CPU (about a minute) and saves `small_cnn.pt`.

Walkthrough: model_compatibility.ipynb \
Converts the CNN, reads the report, compares accuracy, finds where rounding error builds up, and folds batch norm to cut transfers.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

## Usage
Open the notebook:

```bash
jupyter notebook model_compatibility.ipynb
```

Or use the converter on your own model:

```python
import npu_compat

npu_model = npu_compat.convert(your_model, example_input)
print(npu_model.report())          # where every layer runs
y = npu_model(x)                   # forward pass through the toolkit
print(npu_model.compare(x))        # float32 vs bfloat16, layer by layer
```

`convert()` takes two options:

| Option | Default | What it does |
|---|---|---|
| `fold_batchnorm` | `False` | Folds each `BatchNorm2d` into the convolution before it. Fewer NPU calls, less data moved, less rounding. |
| `weight_budget_gb` | `17.0` | How many GB of weights to keep resident on the card. Layers past the budget send their weights with every call. The driver enforces about 17 GB per card on hardware. The CPU backend doesn't, so the tool checks it. |

## Testing
PyTest test case:
The file test_model_compatibility.py trains the CNN, checks that the toolkit run makes the same predictions as float32, checks the fallbacks for unsupported layers and the weight budget, and executes the notebook.

```bash
pytest test_model_compatibility.py
```

# Description

## What the report shows

On the example CNN (toolkit 2.4.0, CPU backend):

```
#   layer    op           runs on         MACs     output      note
1   conv1    Conv2d       NPU             56,448   1x8x28x28
2   bn1      BatchNorm2d  NPU             6,272    1x8x28x28   scale on NPU, normalize and shift on host, one image per call
3   relu     relu         host (toolkit)           1x8x28x28
4   pool     MaxPool2d    host (toolkit)           1x8x14x14   one image per call
...
NPU share of multiply-accumulates   100.0%
NPU calls per forward pass          6
Switches between NPU and host       12
Data to and from the card per pass  116.6 KB
```

With `fold_batchnorm=True`, the same model needs 4 NPU calls, 8 switches and 61.5 KB per pass, with the same predictions.

## Where each operation runs

Taken from the toolkit v2.4.0 source:

| PyTorch | Runs on | Toolkit function |
|---|---|---|
| `nn.Linear` | NPU | `linear_fprop_preloaded_weights`, bias with `add_bias_fprop` on the host |
| `nn.Conv2d` | NPU, after unfolding on the host | `conv_fprop_preloaded_weights` |
| `nn.ConvTranspose2d` | NPU | `conv_transpose_fprop_preloaded_weights` |
| `nn.BatchNorm2d` | Scale on NPU, normalize and shift on host | `batchnorm2d_fprop` |
| ReLU, sigmoid, softmax | Host | `relu_fprop`, `sigmoid_fprop`, `softmax_fprop` |
| Max, average and adaptive pooling | Host | `maxpool2d_fprop`, `avgpool2d_fprop`, `adaptive_*pool2d_fprop` |
| Flatten, view, reshape, dropout in eval | No arithmetic | - |
| Everything else (LayerNorm, GELU, residual adds, attention, ...) | Host, in PyTorch | - |

## Limits
- The model has to be traceable with `torch.fx`.
- Convolutions need `groups=1`, `dilation=1` and the same padding and stride in both directions. Others stay in PyTorch, and the report says why.
- Transposed convolution, pooling, softmax and batch norm reject batched input in toolkit 2.4.0, so the converter calls them one image at a time.
- Byte counts are estimates from the toolkit source, not measurements. On hardware, measure with `get_perf_counter`.
- The CPU backend emulates the arithmetic, so this checks compatibility and accuracy, not speed.
