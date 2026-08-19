import torchvision
import torch
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

torch.manual_seed(42)


def load_random_image():
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "/tmp/mnist_files/",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=None,
        shuffle=True,
    )
    target_img_size = 28
    # torch loader shuffles anyway
    image, label = next(iter(data_loader))
    img_scaled = torchvision.transforms.Resize((target_img_size, target_img_size))(
        image
    ).numpy()
    img_scaled -= np.min(img_scaled)
    img_scaled /= np.max(img_scaled)
    img_scaled = img_scaled.astype(bfloat16)

    return img_scaled, image, label


def load_NN_weights():
    weight_dict = np.load(Path(__file__).resolve().parent / "network_weights.npz")
    weights_layer_1 = weight_dict["l1_w"].astype(bfloat16)
    bias_layer_1 = weight_dict["l1_b"].astype(bfloat16)
    weights_layer_2 = weight_dict["l2_w"].astype(bfloat16)
    bias_layer_2 = weight_dict["l2_b"].astype(bfloat16)
    return weights_layer_1, bias_layer_1, weights_layer_2, bias_layer_2


def display_result(digit, certainty):
    from IPython.display import display, HTML

    display(
        HTML(
            f'<p style="font-size: 48px;">The provided image contains the number: <b>{digit}</b>'
        )
    )
    display(HTML(f'<p style="font-size: 28px;">with a certainty of: {certainty}%</p>'))
    display(
        HTML(
            '<p style="font-size: 28px;">Q.ANT\'s Photonic AI promises 30 times less power consumption than electronics!</p>'
        )
    )
