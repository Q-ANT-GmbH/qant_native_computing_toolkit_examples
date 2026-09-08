import numpy as np
import requests
from PIL import Image
from skimage.exposure import rescale_intensity
from torchvision import transforms


def get_image(url):
    image = np.array(Image.open(requests.get(url, stream=True).raw))

    preprocess = transforms.Compose([normalize_volume, transforms.ToTensor()])

    # Preprocess image
    input_tensor = preprocess(image.astype(np.float32))
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    return image, input_batch


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


def normalize_volume(volume):
    """
    Normalize the image data as done for the training

    See https://github.com/mateuszbuda/brain-segmentation-pytorch

    """
    p10 = np.percentile(volume, 10)
    p99 = np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    m = np.mean(volume, axis=(0, 1, 2))
    s = np.std(volume, axis=(0, 1, 2))
    volume = (volume - m) / s
    return volume
