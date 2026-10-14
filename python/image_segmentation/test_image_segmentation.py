import importlib
import os

import numpy as np
import torch
from testbook import testbook


def compare_masks(mask_a: np.ndarray, mask_b: np.ndarray):
    difference = (np.abs(mask_a == mask_b)).sum()

    mask_a = np.round(mask_a)
    mask_b = np.round(mask_b)

    overlap = (mask_a == mask_b).astype(np.float32).mean()

    return difference, overlap


def test_unet_overlap():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))

    # this has to be loaded in the correct working directory
    # and reloaded as maybe another test already contains a utils
    import utils

    importlib.reload(utils)

    with testbook(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./image_segmentation_on_npu.ipynb",
        ),
        execute=True,
        timeout=1200,
    ) as tb:
        img_url = tb.get("url")
        output_qant = np.array(tb.get("output_list"))

        _, input_batch = utils.get_image(img_url)
        model_torch = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        model_torch.eval()
        with torch.no_grad():
            output_torch = model_torch(input_batch).numpy()

        output_qant.reshape(output_torch.shape)

        print(f"{output_torch.shape = }")
        print(f"{output_qant.shape = }")
        err = np.abs(output_torch - output_qant)
        print(f"Total difference: {np.sum(err)}")
        print(f"Average difference (mean): {np.mean(err)}")
        print(f"Average difference (median): {np.median(err)}")
        print(f"Average value (mean): {np.mean(np.abs(output_torch))}")
        relative_error = np.divide(
            err,
            np.abs(output_torch),
            out=np.zeros_like(output_torch),
            where=output_torch != 0,
        )
        print(f"Average relative difference (mean): {np.mean(relative_error)}")

        _, overlap = compare_masks(output_qant[0, 0], output_torch[0, 0])

        np.testing.assert_(overlap > 0.95)
