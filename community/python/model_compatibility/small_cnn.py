"""A small, ordinary PyTorch CNN for MNIST, written with no knowledge of the NPU.

It stands in for "the model you already have". Training takes about a minute
on a laptop CPU.
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = pathlib.Path("/tmp/mnist_files")
WEIGHTS = pathlib.Path(__file__).with_name("small_cnn.pt")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


def mnist(train: bool) -> datasets.MNIST:
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    return datasets.MNIST(DATA_DIR, train=train, download=True, transform=tf)


def train(epochs: int = 1, seed: int = 0, weights: pathlib.Path = WEIGHTS) -> SmallCNN:
    """Train on CPU and save the weights. Returns the trained model in eval mode."""
    torch.manual_seed(seed)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loader = DataLoader(mnist(train=True), batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    torch.save(model.state_dict(), weights)
    return model.eval()


def load_or_train(weights: pathlib.Path = WEIGHTS) -> SmallCNN:
    if weights.exists():
        model = SmallCNN()
        model.load_state_dict(torch.load(weights, weights_only=True))
        return model.eval()
    return train(weights=weights)


def test_batch(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """n images and labels from the MNIST test set, picked reproducibly."""
    ds = mnist(train=False)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
    xs, ys = zip(*(ds[int(i)] for i in idx))
    return torch.stack(xs), torch.tensor(ys)
