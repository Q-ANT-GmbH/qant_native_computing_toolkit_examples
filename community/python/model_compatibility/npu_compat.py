"""Check and run an existing PyTorch model on the Q.ANT NPU.

Community example. Not maintained or supported by Q.ANT.

The module answers three questions about an unmodified ``torch.nn.Module``:

1. Which parts of it can the toolkit run, and where (NPU or host)?
2. How much data crosses the PCIe link in one forward pass?
3. How far does the bfloat16 result drift from the float32 original,
   overall and layer by layer?

Typical use::

    import npu_compat

    npu_model = npu_compat.convert(model, example_input)
    print(npu_model.report())          # where every layer runs
    y = npu_model(x)                   # forward pass through the toolkit
    print(npu_model.compare(x))        # float32 vs bfloat16, layer by layer

The tool traces the model with ``torch.fx``, so operations written inside
``forward()`` (functional calls, flatten, residual adds) are seen as well as
``nn.Module`` layers. Anything the toolkit cannot run stays in PyTorch on the
host, so every traceable model runs end to end.

Where an operation runs is taken from the toolkit v2.4.0 source:

* NPU: linear and convolution matrix multiplications, and the scale step of
  batch normalization.
* Toolkit on the host CPU: bias, ReLU, sigmoid, softmax, pooling, and the
  normalize and shift steps of batch normalization.

On the CPU backend everything runs on the host, so this tool checks
compatibility and accuracy, not speed.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import qant_native_computing_toolkit.ai as q_ai
import qant_native_computing_toolkit.generic as q_generic
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from ml_dtypes import bfloat16
from torch.fx.passes.shape_prop import ShapeProp

# Where a node runs.
NPU = "npu"  # matrix work on the card
HOST = "toolkit-host"  # toolkit function that runs on the host CPU
TORCH = "pytorch"  # not supported by the toolkit, left in PyTorch on the host
SHAPE = "shape"  # no arithmetic (flatten, view, dropout in eval, ...)

BF16_BYTES = 2
DEFAULT_WEIGHT_BUDGET_GB = 17.0  # rough per-card limit, from the maintainers


# --------------------------------------------------------------------------
# bfloat16 helpers
# --------------------------------------------------------------------------
def _to_bf16(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.detach().float().cpu().numpy().astype(bfloat16))


def _from_bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a).astype(np.float32))


def _pair(v: Any) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _square(v: Any, name: str) -> int:
    """Return v as one int if both directions match, else raise."""
    a, b = _pair(v)
    if a != b:
        raise _Unsupported(f"{name} {a}x{b} differs per direction")
    return int(a)


class _Unsupported(Exception):
    """Raised while planning a node the toolkit cannot run."""


# --------------------------------------------------------------------------
# Report data
# --------------------------------------------------------------------------
@dataclass
class Layer:
    """One traced node and how it is executed."""

    name: str
    op: str
    where: str
    out_shape: tuple[int, ...]
    macs: int = 0  # multiply-accumulates (estimated)
    steps: list[str] = field(default_factory=list)  # NPU/HOST sequence, for switches
    upload_bytes: int = 0  # features sent to the card per forward pass
    download_bytes: int = 0  # results read back per forward pass
    weight_bytes: int = 0  # resident on the card
    weights_resident: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "op": self.op,
            "where": self.where,
            "out_shape": list(self.out_shape),
            "macs": self.macs,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "weight_bytes": self.weight_bytes,
            "weights_resident": self.weights_resident,
            "note": self.note,
            "steps": self.steps,
        }


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_count(n: int) -> str:
    return f"{n:,}"


_WHERE_LABEL = {
    NPU: "NPU",
    HOST: "host (toolkit)",
    TORCH: "host (PyTorch)",
    SHAPE: "-",
}


@dataclass
class Report:
    """Where every layer runs, plus totals for one forward pass."""

    model_name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    weight_budget_bytes: int

    # ---- totals
    @property
    def total_macs(self) -> int:
        return sum(layer.macs for layer in self.layers)

    @property
    def npu_macs(self) -> int:
        return sum(layer.macs for layer in self.layers if layer.where == NPU)

    @property
    def npu_share(self) -> float:
        return self.npu_macs / self.total_macs if self.total_macs else 0.0

    @property
    def npu_calls(self) -> int:
        return sum(layer.steps.count(NPU) for layer in self.layers)

    @property
    def switches(self) -> int:
        """Changes between NPU and host, starting from input on the host."""
        seq = [HOST] + [s for layer in self.layers for s in layer.steps]
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    @property
    def transfer_bytes(self) -> int:
        return sum(layer.upload_bytes + layer.download_bytes for layer in self.layers)

    @property
    def resident_weight_bytes(self) -> int:
        return sum(
            layer.weight_bytes for layer in self.layers if layer.weights_resident
        )

    @property
    def left_in_pytorch(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.where == TORCH]

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            "input_shape": list(self.input_shape),
            "layers": [layer.as_dict() for layer in self.layers],
            "summary": {
                "total_macs": self.total_macs,
                "npu_macs": self.npu_macs,
                "npu_share": self.npu_share,
                "npu_calls": self.npu_calls,
                "switches": self.switches,
                "transfer_bytes": self.transfer_bytes,
                "resident_weight_bytes": self.resident_weight_bytes,
                "weight_budget_bytes": self.weight_budget_bytes,
            },
        }

    def __str__(self) -> str:
        rows = [("#", "layer", "op", "runs on", "MACs", "output", "note")]
        for i, layer in enumerate(self.layers, 1):
            rows.append(
                (
                    str(i),
                    layer.name,
                    layer.op,
                    _WHERE_LABEL[layer.where],
                    _fmt_count(layer.macs) if layer.macs else "",
                    "x".join(map(str, layer.out_shape)),
                    layer.note,
                )
            )
        widths = [max(len(r[c]) for r in rows) for c in range(6)]
        lines = [f"{self.model_name}, input {'x'.join(map(str, self.input_shape))}", ""]
        for r in rows:
            cells = [r[c].ljust(widths[c]) for c in range(6)]
            lines.append("  ".join(cells) + (f"  {r[6]}" if r[6] else ""))
        torch_ops = sorted({layer.op for layer in self.left_in_pytorch})
        lines += [
            "",
            f"NPU share of multiply-accumulates   {self.npu_share:.1%}",
            f"NPU calls per forward pass          {self.npu_calls}",
            f"Switches between NPU and host       {self.switches}",
            f"Data to and from the card per pass  {_fmt_bytes(self.transfer_bytes)}",
            f"Weights resident on the card        {_fmt_bytes(self.resident_weight_bytes)}"
            f" of {_fmt_bytes(self.weight_budget_bytes)} budget",
            f"Left in PyTorch                     {', '.join(torch_ops) if torch_ops else 'nothing'}",
        ]
        return "\n".join(lines)


@dataclass
class LayerError:
    name: str
    op: str
    where: str
    accumulated: float  # relative error of this layer's output in the full bf16 run
    local: float | None  # relative error added by this layer alone (float32 inputs)


@dataclass
class Comparison:
    """float32 original vs the toolkit run, overall and per layer."""

    output_error: float
    top1_agreement: float | None
    layers: list[LayerError]

    def as_dict(self) -> dict:
        return {
            "output_error": self.output_error,
            "top1_agreement": self.top1_agreement,
            "layers": [vars(e) for e in self.layers],
        }

    def __str__(self) -> str:
        w = max([len("layer")] + [len(e.name) for e in self.layers]) + 2
        lines = [f"{'layer':<{w}}{'runs on':<16}{'added here':>12}{'so far':>10}"]
        for e in self.layers:
            local = "" if e.local is None else f"{e.local:.2%}"
            where = _WHERE_LABEL[e.where]
            lines.append(f"{e.name:<{w}}{where:<16}{local:>12}{e.accumulated:>10.2%}")
        lines.append("")
        lines.append(f"Relative error of the final output  {self.output_error:.2%}")
        if self.top1_agreement is not None:
            lines.append(
                f"Same top-1 prediction as float32    {self.top1_agreement:.1%}"
            )
        return "\n".join(lines)


def relative_error(y: torch.Tensor, ref: torch.Tensor) -> float:
    """||y - ref|| / ||ref||, the error measure used throughout."""
    ref = ref.detach().float()
    denom = ref.norm().item()
    diff = (y.detach().float() - ref).norm().item()
    return diff / denom if denom else diff


# --------------------------------------------------------------------------
# Planning: decide where each node runs and build a runner for it
# --------------------------------------------------------------------------
Runner = Callable[..., torch.Tensor]


@dataclass
class _Plan:
    layer: Layer
    runner: Runner | None  # None: run the node with PyTorch


def _out_shape(node: fx.Node) -> tuple[int, ...]:
    meta = node.meta.get("tensor_meta")
    return tuple(meta.shape) if meta is not None and hasattr(meta, "shape") else ()


def _in_shape(node: fx.Node) -> tuple[int, ...]:
    arg = node.args[0] if node.args else None
    return _out_shape(arg) if isinstance(arg, fx.Node) else ()


def _numel(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape)) if shape else 0


class _Budget:
    """Tracks how many weight bytes are kept resident on the card."""

    def __init__(self, limit_bytes: int):
        self.limit = limit_bytes
        self.used = 0

    def take(self, nbytes: int) -> bool:
        if self.used + nbytes > self.limit:
            return False
        self.used += nbytes
        return True


def _plan_linear(m: nn.Linear, node: fx.Node, budget: _Budget, dev: int) -> _Plan:
    shape_in, shape_out = _in_shape(node), _out_shape(node)
    w = _to_bf16(m.weight)
    b = _to_bf16(m.bias) if m.bias is not None else None
    wbytes = w.size * BF16_BYTES
    resident = budget.take(wbytes)
    handle = (
        q_generic.upload(w, q_generic.DataRole.Weights, device_id=dev)
        if resident
        else None
    )

    def run(x: torch.Tensor) -> torch.Tensor:
        f = _to_bf16(x)
        if handle is not None:
            y = q_ai.linear_fprop_preloaded_weights(f, handle)
        else:
            y = q_ai.linear_fprop(f, w, device_id=dev)
        if b is not None:
            flat = y.reshape(-1, y.shape[-1])
            y = q_ai.add_bias_fprop(flat, b, device_id=dev).reshape(y.shape)
        return _from_bf16(y)

    layer = Layer(
        name=node.name,
        op="Linear",
        where=NPU,
        out_shape=shape_out,
        macs=_numel(shape_out[:-1]) * m.in_features * m.out_features,
        steps=[NPU] + ([HOST] if b is not None else []),
        upload_bytes=_numel(shape_in) * BF16_BYTES + (0 if resident else wbytes),
        download_bytes=_numel(shape_out) * BF16_BYTES,
        weight_bytes=wbytes,
        weights_resident=resident,
        note="" if resident else "weights over budget, sent every call",
    )
    return _Plan(layer, run)


def _conv_params(m: nn.Conv2d | nn.ConvTranspose2d) -> tuple[int, int]:
    if m.groups != 1:
        raise _Unsupported(f"groups={m.groups}")
    if _pair(m.dilation) != (1, 1):
        raise _Unsupported("dilation other than 1")
    if m.padding_mode != "zeros":
        raise _Unsupported(f"padding_mode={m.padding_mode}")
    if isinstance(m.padding, str):
        if m.padding == "valid":
            return 0, _square(m.stride, "stride")
        raise _Unsupported("padding='same'")
    return _square(m.padding, "padding"), _square(m.stride, "stride")


def _plan_conv(m: nn.Conv2d, node: fx.Node, budget: _Budget, dev: int) -> _Plan:
    padding, stride = _conv_params(m)
    shape_in, shape_out = _in_shape(node), _out_shape(node)
    if len(shape_in) != 4:
        raise _Unsupported("needs batched 4D input")
    w = _to_bf16(m.weight)
    b = _to_bf16(m.bias) if m.bias is not None else None
    wbytes = w.size * BF16_BYTES
    resident = budget.take(wbytes)
    handle = q_ai.upload_conv_weights(w, device_id=dev) if resident else None

    def run(x: torch.Tensor) -> torch.Tensor:
        f = _to_bf16(x)
        if handle is not None:
            y = q_ai.conv_fprop_preloaded_weights(f, handle, padding, stride, 1)
        else:
            y = q_ai.conv_fprop(f, w, padding, stride, 1, device_id=dev)
        if b is not None:
            y = q_ai.add_bias_fprop(y, b, device_id=dev)
        return _from_bf16(y)

    n, _, h_out, w_out = shape_out
    kh, kw = _pair(m.kernel_size)
    patches = n * h_out * w_out * m.in_channels * kh * kw  # unfolded on the host first
    layer = Layer(
        name=node.name,
        op="Conv2d",
        where=NPU,
        out_shape=shape_out,
        macs=_numel(shape_out) * m.in_channels * kh * kw,
        steps=[HOST, NPU] + ([HOST] if b is not None else []),
        upload_bytes=patches * BF16_BYTES + (0 if resident else wbytes),
        download_bytes=_numel(shape_out) * BF16_BYTES,
        weight_bytes=wbytes,
        weights_resident=resident,
        note="" if resident else "weights over budget, sent every call",
    )
    return _Plan(layer, run)


def _plan_conv_transpose(
    m: nn.ConvTranspose2d, node: fx.Node, budget: _Budget, dev: int
) -> _Plan:
    padding, stride = _conv_params(m)
    out_pad = _square(m.output_padding, "output_padding")
    shape_in, shape_out = _in_shape(node), _out_shape(node)
    if len(shape_in) != 4:
        raise _Unsupported("needs batched 4D input")
    w = _to_bf16(m.weight)
    b = _to_bf16(m.bias) if m.bias is not None else None
    wbytes = w.size * BF16_BYTES
    resident = budget.take(wbytes)
    handle = q_ai.upload_conv_transpose_weights(w, device_id=dev) if resident else None

    def run(x: torch.Tensor) -> torch.Tensor:
        def one(f: np.ndarray) -> np.ndarray:
            if handle is not None:
                return q_ai.conv_transpose_fprop_preloaded_weights(
                    f, handle, padding, stride, 1, out_pad
                )
            return q_ai.conv_transpose_fprop(
                f, w, padding, stride, 1, out_pad, device_id=dev
            )

        y = _per_image(one, _to_bf16(x))
        if b is not None:
            y = q_ai.add_bias_fprop(y, b, device_id=dev)
        return _from_bf16(y)

    kh, kw = _pair(m.kernel_size)
    layer = Layer(
        name=node.name,
        op="ConvTranspose2d",
        where=NPU,
        out_shape=shape_out,
        macs=_numel(shape_in) * m.out_channels * kh * kw,
        steps=[HOST, NPU] * shape_in[0] + ([HOST] if b is not None else []),
        upload_bytes=_numel(shape_in) * BF16_BYTES + (0 if resident else wbytes),
        download_bytes=_numel(shape_out) * BF16_BYTES,
        weight_bytes=wbytes,
        weights_resident=resident,
        note="one image per call, transfer is a lower bound"
        + ("" if resident else ", weights sent every call"),
    )
    return _Plan(layer, run)


def _plan_batchnorm(m: nn.BatchNorm2d, node: fx.Node, dev: int) -> _Plan:
    if not (m.affine and m.track_running_stats):
        raise _Unsupported("needs affine=True and running statistics")
    shape_out = _out_shape(node)
    mean, var = _to_bf16(m.running_mean), _to_bf16(m.running_var)
    weight, bias = _to_bf16(m.weight), _to_bf16(m.bias)
    eps = float(m.eps)

    def run(x: torch.Tensor) -> torch.Tensor:
        f = _to_bf16(x)
        return _from_bf16(
            _per_image(
                lambda one: q_ai.batchnorm2d_fprop(
                    one, mean, var, weight, bias, eps, device_id=dev
                ),
                f,
            )
        )

    batch = shape_out[0] if shape_out else 1
    numel = _numel(shape_out)
    return _Plan(
        Layer(
            name=node.name,
            op="BatchNorm2d",
            where=NPU,
            out_shape=shape_out,
            macs=numel,
            steps=[HOST, NPU, HOST] * batch,
            upload_bytes=2 * numel * BF16_BYTES,  # normalized values and expanded scale
            download_bytes=numel * BF16_BYTES,
            note="scale on NPU, normalize and shift on host, one image per call",
        ),
        run,
    )


def _per_image(fn: Callable[[np.ndarray], np.ndarray], f: np.ndarray) -> np.ndarray:
    """Call fn on one item at a time, for toolkit functions that reject batches."""
    if f.shape[0] == 1:
        return fn(f)
    return np.concatenate([fn(f[i : i + 1]) for i in range(f.shape[0])])


def _host(
    node: fx.Node, op: str, fn: Callable[[np.ndarray], np.ndarray], batched: bool = True
) -> _Plan:
    """A toolkit function that runs on the host CPU."""

    def run(x: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        f = _to_bf16(x)
        return _from_bf16(fn(f) if batched else _per_image(fn, f))

    note = "" if batched else "one image per call"
    return _Plan(
        Layer(node.name, op, HOST, _out_shape(node), steps=[HOST], note=note), run
    )


def _softmax(dev: int) -> Callable[[np.ndarray], np.ndarray]:
    """Softmax over the last dim, one row per call (the toolkit rejects batches)."""

    def fn(f: np.ndarray) -> np.ndarray:
        rows = f.reshape(-1, f.shape[-1])
        return _per_image(lambda r: q_ai.softmax_fprop(r, device_id=dev), rows).reshape(
            f.shape
        )

    return fn


def _pool_params(
    kernel: Any, stride: Any, padding: Any
) -> tuple[tuple[int, int], int, int]:
    kernel = _pair(kernel)
    stride = _square(stride if stride not in (None, []) else kernel, "stride")
    return kernel, stride, _square(padding, "padding")


def _plan_maxpool(
    node: fx.Node, kernel, stride, padding, dilation, ceil_mode, return_indices, dev
) -> _Plan:
    if _pair(dilation) != (1, 1) or ceil_mode or return_indices:
        raise _Unsupported("dilation, ceil_mode or return_indices")
    k, s, p = _pool_params(kernel, stride, padding)
    return _host(
        node,
        "MaxPool2d",
        lambda f: q_ai.maxpool2d_fprop(f, k, s, p, device_id=dev),
        batched=False,
    )


def _plan_avgpool(
    node: fx.Node,
    kernel,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dev,
) -> _Plan:
    if ceil_mode or divisor_override is not None:
        raise _Unsupported("ceil_mode or divisor_override")
    k, s, p = _pool_params(kernel, stride, padding)
    return _host(
        node,
        "AvgPool2d",
        lambda f: q_ai.avgpool2d_fprop(f, k, s, p, count_include_pad, device_id=dev),
        batched=False,
    )


def _plan_adaptive(node: fx.Node, output_size: Any, kind: str, dev: int) -> _Plan:
    shape_in = _in_shape(node)
    oh, ow = _pair(output_size)
    if (
        None in (oh, ow)
        or shape_in[-1] % ow
        or shape_in[-2] % oh
        or oh != ow
        or shape_in[-1] != shape_in[-2]
    ):
        raise _Unsupported(
            "needs square input that is a whole multiple of the output size"
        )
    fn = (
        q_ai.adaptive_avgpool2d_fprop
        if kind == "avg"
        else q_ai.adaptive_maxpool2d_fprop
    )
    return _host(
        node,
        f"Adaptive{kind.capitalize()}Pool2d",
        lambda f: fn(f, (oh, ow), device_id=dev),
        batched=False,
    )


def _softmax_dim_ok(node: fx.Node, dim: Any) -> None:
    ndim = len(_out_shape(node))
    if dim not in (-1, ndim - 1):
        raise _Unsupported(f"softmax over dim {dim}, toolkit uses the last dim")


_SHAPE_FUNCS = {
    torch.flatten,
    torch.reshape,
    torch.squeeze,
    torch.unsqueeze,
    operator.getitem,
    getattr,
}
_SHAPE_METHODS = {
    "view",
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
    "size",
    "contiguous",
    "permute",
    "transpose",
}
_SHAPE_MODULES = (nn.Flatten, nn.Identity, nn.Dropout, nn.Dropout2d)


def _arg(node: fx.Node, i: int, key: str, default: Any) -> Any:
    if len(node.args) > i:
        return node.args[i]
    return node.kwargs.get(key, default)


def _plan_node(
    node: fx.Node, gm: fx.GraphModule, budget: _Budget, dev: int
) -> _Plan | None:
    """Return a plan for a compute node, or None for placeholders and outputs."""
    if node.op in ("placeholder", "output", "get_attr"):
        return None

    if node.op == "call_module":
        m = gm.get_submodule(node.target)
        op = type(m).__name__
        if isinstance(m, nn.Linear):
            return _plan_linear(m, node, budget, dev)
        if isinstance(m, nn.Conv2d):
            return _plan_conv(m, node, budget, dev)
        if isinstance(m, nn.ConvTranspose2d):
            return _plan_conv_transpose(m, node, budget, dev)
        if isinstance(m, nn.BatchNorm2d):
            return _plan_batchnorm(m, node, dev)
        if isinstance(m, nn.ReLU):
            return _host(node, op, lambda f: q_ai.relu_fprop(f, device_id=dev))
        if isinstance(m, nn.Sigmoid):
            return _host(node, op, lambda f: q_ai.sigmoid_fprop(f, device_id=dev))
        if isinstance(m, nn.Softmax):
            _softmax_dim_ok(node, m.dim)
            return _host(node, op, _softmax(dev))
        if isinstance(m, nn.MaxPool2d):
            return _plan_maxpool(
                node,
                m.kernel_size,
                m.stride,
                m.padding,
                m.dilation,
                m.ceil_mode,
                m.return_indices,
                dev,
            )
        if isinstance(m, nn.AvgPool2d):
            return _plan_avgpool(
                node,
                m.kernel_size,
                m.stride,
                m.padding,
                m.ceil_mode,
                m.count_include_pad,
                m.divisor_override,
                dev,
            )
        if isinstance(m, nn.AdaptiveAvgPool2d):
            return _plan_adaptive(node, m.output_size, "avg", dev)
        if isinstance(m, nn.AdaptiveMaxPool2d):
            if m.return_indices:
                raise _Unsupported("return_indices")
            return _plan_adaptive(node, m.output_size, "max", dev)
        if isinstance(m, _SHAPE_MODULES):
            return _Plan(Layer(node.name, op, SHAPE, _out_shape(node)), None)
        raise _Unsupported(f"{op} has no toolkit function")

    if node.op == "call_function":
        fn = node.target
        name = getattr(fn, "__name__", str(fn))
        if fn in (F.relu, torch.relu):
            return _host(node, "relu", lambda f: q_ai.relu_fprop(f, device_id=dev))
        if fn in (torch.sigmoid, F.sigmoid):
            return _host(
                node, "sigmoid", lambda f: q_ai.sigmoid_fprop(f, device_id=dev)
            )
        if fn in (F.softmax, torch.softmax):
            _softmax_dim_ok(node, _arg(node, 1, "dim", None))
            return _host(node, "softmax", _softmax(dev))
        if fn is F.max_pool2d:
            return _plan_maxpool(
                node,
                _arg(node, 1, "kernel_size", None),
                _arg(node, 2, "stride", None),
                _arg(node, 3, "padding", 0),
                _arg(node, 4, "dilation", 1),
                _arg(node, 5, "ceil_mode", False),
                _arg(node, 6, "return_indices", False),
                dev,
            )
        if fn is F.avg_pool2d:
            return _plan_avgpool(
                node,
                _arg(node, 1, "kernel_size", None),
                _arg(node, 2, "stride", None),
                _arg(node, 3, "padding", 0),
                _arg(node, 4, "ceil_mode", False),
                _arg(node, 5, "count_include_pad", True),
                _arg(node, 6, "divisor_override", None),
                dev,
            )
        if fn is F.adaptive_avg_pool2d:
            return _plan_adaptive(node, _arg(node, 1, "output_size", None), "avg", dev)
        if fn in _SHAPE_FUNCS or (
            fn is F.dropout and not _arg(node, 2, "training", False)
        ):
            return _Plan(Layer(node.name, name, SHAPE, _out_shape(node)), None)
        raise _Unsupported(f"{name} has no toolkit function")

    if node.op == "call_method":
        if node.target in _SHAPE_METHODS:
            return _Plan(
                Layer(node.name, f".{node.target}", SHAPE, _out_shape(node)), None
            )
        if node.target == "relu":
            return _host(node, ".relu", lambda f: q_ai.relu_fprop(f, device_id=dev))
        raise _Unsupported(f".{node.target}() has no toolkit function")

    raise _Unsupported(node.op)


def _torch_macs(node: fx.Node, gm: fx.GraphModule) -> int:
    """Rough MAC count for layers left in PyTorch, so the NPU share stays honest."""
    out = _out_shape(node)
    if node.op == "call_module":
        m = gm.get_submodule(node.target)
        if isinstance(m, nn.Linear):
            return _numel(out[:-1]) * m.in_features * m.out_features
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            k = int(np.prod(m.kernel_size))
            return _numel(out) * (m.in_channels // m.groups) * k
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            return _numel(out)
        if isinstance(m, nn.MultiheadAttention):
            e = m.embed_dim
            seq = _in_shape(node)
            n_tok = _numel(seq[:-1])
            return 4 * n_tok * e * e + 2 * n_tok * (seq[-2] if len(seq) > 1 else 1) * e
    if node.op == "call_function" and node.target in (
        torch.matmul,
        operator.matmul,
        torch.bmm,
    ):
        a = _in_shape(node)
        return _numel(out) * (a[-1] if a else 0)
    return 0


# --------------------------------------------------------------------------
# The converted model
# --------------------------------------------------------------------------
class _Interpreter(fx.Interpreter):
    """Runs the traced graph, sending planned nodes to the toolkit."""

    def __init__(
        self,
        gm: fx.GraphModule,
        plans: dict[str, _Plan] | None,
        record: dict | None = None,
    ):
        super().__init__(gm)
        self.plans = plans or {}
        self.record = record

    def run_node(self, n: fx.Node) -> Any:
        plan = self.plans.get(n.name)
        if plan is not None and plan.runner is not None:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            out = plan.runner(*args)
        else:
            out = super().run_node(n)
        if self.record is not None and isinstance(out, torch.Tensor):
            self.record[n.name] = out
        return out


class NPUModel:
    """A PyTorch model whose supported layers run through the Q.ANT toolkit."""

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        device_id: int = 0,
        weight_budget_gb: float = DEFAULT_WEIGHT_BUDGET_GB,
        fold_batchnorm: bool = False,
    ):
        model = model.eval()
        gm = fx.symbolic_trace(model)
        if fold_batchnorm:
            from torch.fx.experimental.optimization import fuse

            gm = fuse(gm, inplace=False)
        with torch.no_grad():
            ShapeProp(gm).propagate(example_input)
        self.gm = gm
        self.model_name = type(model).__name__ + (
            " (batch norm folded)" if fold_batchnorm else ""
        )
        self.input_shape = tuple(example_input.shape)
        budget = _Budget(int(weight_budget_gb * 1024**3))
        self.plans: dict[str, _Plan] = {}
        for node in gm.graph.nodes:
            try:
                plan = _plan_node(node, gm, budget, device_id)
            except _Unsupported as why:
                op = (
                    type(gm.get_submodule(node.target)).__name__
                    if node.op == "call_module"
                    else getattr(node.target, "__name__", str(node.target))
                )
                plan = _Plan(
                    Layer(
                        node.name,
                        op,
                        TORCH,
                        _out_shape(node),
                        macs=_torch_macs(node, gm),
                        steps=[HOST],
                        note=str(why),
                    ),
                    None,
                )
            if plan is not None:
                self.plans[node.name] = plan
        self._budget_bytes = budget.limit

    # ---- running
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _Interpreter(self.gm, self.plans).run(x)

    # ---- inspecting
    def report(self) -> Report:
        return Report(
            self.model_name,
            self.input_shape,
            [p.layer for p in self.plans.values()],
            self._budget_bytes,
        )

    @torch.no_grad()
    def compare(self, x: torch.Tensor) -> Comparison:
        """Compare against float32 PyTorch, overall and layer by layer.

        ``so far`` is the error of each layer's output in the full toolkit run,
        so it includes everything upstream. ``added here`` feeds the layer the
        exact float32 input and measures only its own rounding.
        """
        ref: dict[str, torch.Tensor] = {}
        ref_env_out = _Interpreter(self.gm, None, ref).run(x)
        got: dict[str, torch.Tensor] = {}
        out = _Interpreter(self.gm, self.plans, got).run(x)

        nodes = {n.name: n for n in self.gm.graph.nodes}
        layers = []
        for name, plan in self.plans.items():
            if plan.layer.where == SHAPE or name not in ref:
                continue
            local = None
            if plan.runner is not None:
                args = fx.node.map_arg(nodes[name].args, lambda a: ref[a.name])
                local = relative_error(plan.runner(*args), ref[name])
            layers.append(
                LayerError(
                    name,
                    plan.layer.op,
                    plan.layer.where,
                    relative_error(got[name], ref[name]),
                    local,
                )
            )

        agree = None
        if ref_env_out.ndim == 2 and ref_env_out.shape[-1] > 1:
            agree = (ref_env_out.argmax(-1) == out.argmax(-1)).float().mean().item()
        return Comparison(relative_error(out, ref_env_out), agree, layers)


def convert(model: nn.Module, example_input: torch.Tensor, **kwargs: Any) -> NPUModel:
    """Trace ``model`` and prepare it to run through the toolkit.

    Args:
        model: Any ``torch.fx``-traceable module. It is put in eval mode.
        example_input: A tensor with the shape you will run, used to trace shapes.
        device_id: NPU to use (0 on the CPU backend).
        weight_budget_gb: How many GB of weights to keep resident on the card.
            Layers beyond the budget send their weights with every call.
        fold_batchnorm: Fold each BatchNorm2d into the convolution before it.
            Fewer NPU-host switches and less bfloat16 rounding.
    """
    return NPUModel(model, example_input, **kwargs)


def export_json(npu_model: NPUModel, x: torch.Tensor, path: str) -> dict:
    """Write the report and the comparison for ``x`` to a JSON file and return them."""
    import json

    data = {
        "toolkit": "qant_native_computing_toolkit 2.4.0",
        "report": npu_model.report().as_dict(),
        "comparison": npu_model.compare(x).as_dict(),
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    return data
