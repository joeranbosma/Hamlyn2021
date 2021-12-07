import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple


def read_input_image(location: str, normalise: bool = True) -> np.ndarray:
    """
    Read input image, stored as png image at the specified location.
    :param location: str, path to input image.

    Returns:
    - input image, shape: (height, width, channels)
    """
    img = cv2.imread(location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if normalise:
        # normalise image
        img = img / 256.

    return img


def read_depth_map(location: str, normalise=True) -> np.ndarray:
    """
    Read input image, stored as png image at the specified location.
    :param location: str, path to input image.

    Returns:
    - input image, shape: (height, width, channels)
    """
    # IMREAD_ANYDEPTH is needed because even though the data is stored in 8-bit channels
    # when it's read into memory it's represented at a higher bit depth
    lbl = cv2.imread(location, flags=cv2.IMREAD_ANYDEPTH)

    if normalise:
        # normalise depth map
        lbl = lbl / 256.

    return lbl


def read_input_sequence(location: str, shape=(256, 512, 3), normalise=True) -> np.ndarray:
    """
    Read input images, stored as png images in the specified folder.
    :param location: str, path to input images.
    :param shape: tuple of ints, dimensions of each input image.

    Returns:
    - input data, shape: (num. timepoints, height, width, channels)
    """
    files = sorted(os.listdir(location))
    files = [fn for fn in files
             if ".png" in fn and "._" not in fn]

    # initialise sequence data
    num = len(files)
    data = np.zeros(shape=(num, *shape))
    for i, fn in enumerate(files):
        img_path = os.path.join(location, fn)
        data[i] = read_input_image(img_path, normalise=normalise)

    return data


def read_depth_sequence(location: str, shape=(256, 512), normalise=True) -> np.ndarray:
    """
    Read depth maps, stored as exr (HDR) images in the specified folder.
    :param location: str, path to input images.
    :param shape: tuple of ints, dimensions of each depth map.

    Returns:
    - depth data, shape: (num. timepoints, height, width)
    """
    files = sorted(os.listdir(location))
    files = [fn for fn in files
             if ".exr" in fn and "._" not in fn]

    # initialise sequence data
    num = len(files)
    data = np.zeros(shape=(num, *shape))
    for i, fn in enumerate(files):
        hdr_path = os.path.join(location, fn)
        data[i] = read_depth_map(hdr_path, normalise=normalise)

    return data


def read_sequence(input_dir: str,
                  depth_dir: str,
                  input_shape=(256, 512, 3),
                  depth_shape=(256, 512)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read input images and depth maps, stored as png/exr images in the specified folders.
    :param input_dir: str, path to input images.
    :param depth_dir: str, path to depth maps.
    :param input_shape: tuple of ints, dimensions of each input image.
    :param depth_shape: tuple of ints, dimensions of each depth map.

    Returns:
    - input data, shape: (num. timepoints, height, width, channels)
    - depth data, shape: (num. timepoints, height, width)
    """
    images = read_input_sequence(location=input_dir, shape=input_shape)
    labels = read_depth_sequence(location=depth_dir, shape=depth_shape)

    return images, labels


if __name__ == "__main__":
    # show example sequence
    print(os.listdir("."))
    input_dir = "/Users/joeranbosma/Hamlyn2021/data/translation_sequences/sequences/scene_1/translation"
    depth_dir = "/Users/joeranbosma/Hamlyn2021/data/depth_sequences/sequences/scene_1/depth"
    images, labels = read_sequence(
        input_dir=input_dir,
        depth_dir=depth_dir,
    )
    print(f"Shape of input images: {images.shape}, shape of depth maps: {labels.shape}")

    for img, lbl in zip(images, labels):
        f, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax = axes[0]
        ax.imshow(img)
        ax = axes[1]
        ax.imshow(lbl)
        plt.show()
        break  # prevent showing a popup for all timesteps
