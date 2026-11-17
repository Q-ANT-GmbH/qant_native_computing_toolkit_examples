import pathlib

import npu_compat
import numpy as np
import pytest
import small_cnn
import testbook
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    weights = tmp_path_factory.mktemp("weights") / "small_cnn.pt"
    model = small_cnn.train(weights=weights)
    images, labels = small_cnn.test_batch(500)
    return model, images, labels


def test_small_cnn_runs_on_toolkit_with_same_predictions(trained):
    model, images, labels = trained
    npu_model = npu_compat.convert(model, images[:1])

    report = npu_model.report()
    assert report.left_in_pytorch == []
    assert report.npu_share > 0.99
    assert report.npu_calls == 6  # 2 conv, 2 batch norm, 2 linear

    with torch.no_grad():
        fp32 = model(images)
    npu = npu_model(images)
    accuracy = (npu.argmax(1) == labels).float().mean().item()
    agreement = (npu.argmax(1) == fp32.argmax(1)).float().mean().item()
    np.testing.assert_array_less(0.95, accuracy)
    np.testing.assert_array_less(0.98, agreement)

    comparison = npu_model.compare(images[:16])
    np.testing.assert_array_less(comparison.output_error, 0.02)
    assert all(e.local is not None for e in comparison.layers)


def test_folding_batchnorm_reduces_trips(trained):
    model, images, _ = trained
    plain = npu_compat.convert(model, images[:1]).report()
    folded_model = npu_compat.convert(model, images[:1], fold_batchnorm=True)
    folded = folded_model.report()

    assert folded.npu_calls < plain.npu_calls
    assert folded.switches < plain.switches
    assert folded.transfer_bytes < plain.transfer_bytes
    with torch.no_grad():
        ref = model(images[:64])
    agree = (folded_model(images[:64]).argmax(1) == ref.argmax(1)).float().mean()
    np.testing.assert_array_less(0.98, agree.item())


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        x = self.inp(x)
        x = x + self.fc(F.gelu(self.norm(x)))
        return F.softmax(self.head(x), dim=-1)


def test_unsupported_layers_stay_in_pytorch():
    torch.manual_seed(0)
    model, x = _Mixer().eval(), torch.randn(4, 16)
    npu_model = npu_compat.convert(model, x[:1])
    left = {layer.op for layer in npu_model.report().left_in_pytorch}
    assert left == {"LayerNorm", "gelu", "add"}

    with torch.no_grad():
        ref = model(x)
    np.testing.assert_allclose(npu_model(x).numpy(), ref.numpy(), atol=2e-2)


def test_conv_options_the_toolkit_lacks_fall_back():
    model = nn.Sequential(
        nn.Conv2d(3, 6, 3, groups=3),
        nn.Conv2d(6, 4, 3, dilation=2),
        nn.Conv2d(4, 4, 3, padding=1),
    ).eval()
    x = torch.randn(2, 3, 16, 16)
    layers = npu_compat.convert(model, x[:1]).report().layers
    assert [layer.where for layer in layers] == [
        npu_compat.TORCH,
        npu_compat.TORCH,
        npu_compat.NPU,
    ]
    assert "groups" in layers[0].note and "dilation" in layers[1].note


def test_weight_budget_streams_layers_that_do_not_fit():
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).eval()
    x = torch.randn(1, 64)
    one_layer_gb = 64 * 64 * 2 / 1024**3
    report = npu_compat.convert(model, x, weight_budget_gb=one_layer_gb).report()
    resident = [
        layer.weights_resident for layer in report.layers if layer.op == "Linear"
    ]
    assert resident == [True, False]
    with torch.no_grad():
        ref = model(x)
    out = npu_compat.convert(model, x, weight_budget_gb=one_layer_gb)(x)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=5e-2)


def test_notebook_runs():
    with testbook.testbook(
        HERE / "model_compatibility.ipynb", execute=True, timeout=600
    ):
        pass
